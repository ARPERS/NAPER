import torch
import torch.nn as nn
import copy
import time
import os
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetNAPER(nn.Module):
    
    def __init__(self, models,  deltas=None, summs=None, output_class=10):
        
        super(NetNAPER, self).__init__()
        self.models = copy.deepcopy(models)
        self.output_class = output_class
        self.protected = False
        self.debug = False
        self.deltas = deltas
        self.summ = summs

        self.N = len(self.models)
        self.wec_time = 0 # estimating inference time of faulty submodel protected

        self.N = len(self.models)
        for i in range(self.N):
            reg_counter = 0
            module_counter = 0
            for name, module in self.models[i].named_modules():
                module_counter += 1
                if len(list(module.children()))==0:
                    module.__name__ = name.replace(".", "[dot]")
                    module.__idx__ = reg_counter  # layer id
                    module.__ens__ = i            # ens id
                    module.__backup__ = 0 if i!=0 else 1 # backup pointer, star topology
                    module.__deltaid__ = max(i, 1) # model_0 delta 1, model 1 delta 1, model 2 delta 2, ...
                    reg_counter += 1
                    module.register_forward_pre_hook(self.csm())
        
        self.n_modules = module_counter
        self.checked = [[False]*self.n_modules for _ in range(self.N)] 
        self.errors_layers = [0]*self.n_modules 
        
        self._clear_errors()
        self._reset_models()

    def csm(self):
        def hook(module, inp, single_check=False):
            with torch.no_grad():
                
                if self.protected:
                    module_id = module.__idx__
                    backup_id = module.__backup__
                    ens_id = module.__ens__
                    delta_id = module.__deltaid__
                    
                    if ens_id!=0 and ((self.checked[ens_id][module_id] and self.errors_layers[module_id]==0) or \
                                      (self.checked[ens_id][module_id] and module_id in self.errors[0])):
                        return

                    backup_module = self.models[backup_id].get_submodule(module.__name__.replace("[dot]","."))
                    delta = self.deltas[delta_id][module_id]
                    sum_new = None

                 
                    for i, ((n, p_0), p_1) in enumerate(zip(module.named_parameters(), backup_module.parameters())):
                        if (ens_id!=0 and self.checked[ens_id][module_id]) \
                            or not torch.allclose(torch.add(p_0, p_1), delta[i]):

                            self.errors_layers[module_id]+=1
                            if sum_new==None:
                                sum_new = sum_module(module)

                            sum_0 = sum_new[i]
                            sum_1 = self.summ[ens_id][module_id][i]

                            # checksum check current model
                            if not torch.isclose(sum_0, sum_1, rtol=1e-08):
                                self.errors[ens_id].append(module_id) 
                                self.errors_meta.append((ens_id, backup_id, delta_id, module.__name__, n, module_id, i))
                            elif ens_id!=0  and module_id not in self.errors[0] and self.checked[0][module_id] \
                                            and module_id not in self.errors[1]:
                                self.deltas[delta_id][module_id][i] = torch.add(p_0, p_1)

                    self.checked[backup_id][module_id] = True

                    if self.errors[ens_id] and single_check==False:
                        for mod in self.models[ens_id].modules():
                            if len(list(mod.children()))==0:
                                if mod.__idx__ > module.__idx__:
                                    hook(mod, inp, single_check=True)
                        raise ValueError("Errors in model")
        return hook

    def _reset_models(self):
        self.next_model = 0 # next model to be run
        self.out_temp = [] # temporary output
        self.used_model = [] # used model for inference

    def _clear_errors(self):
        self.errors = [[] for _ in range(self.N)]
        self.errors_meta = [] 

    def recover_recent(self):
        self.recover(
                self.errors_meta[0][0], # erronous ensemble id (__ens__)
                self.errors_meta[0][1], # backup id (__backup__)
                self.errors_meta[0][2], # deltaid (__deltaid__)
                self.errors_meta[0][3], # erronous module name (__name__)
                self.errors_meta[0][4], # param name n
                self.errors_meta[0][5], # module id (__idx__)
                self.errors_meta[0][6]  # parameter id
            )

    def recover(self, ensid, backup_id, deltaid, modulename, paramname, module_id, param_id):
        with torch.no_grad():
            delta = self.deltas[deltaid][module_id][param_id]
            p_0 = self.models[ensid].get_parameter(modulename.replace("[dot]",".")+"."+paramname)
            p_1 = self.models[backup_id].get_parameter(modulename.replace("[dot]",".")+"."+paramname)
            p_0.copy_(torch.sub(delta, p_1))
            self.errors[ensid].pop(0)
            self.errors_meta.pop(0)
            self.errors_layers[module_id]-=1
            if self.errors_layers[module_id]==0:
                self.checked[ensid][module_id]=False

    def cont_forward(self, x, available_time=1e5, next_model=2):
        self.next_model = next_model
        while True:
            if self.wec_time < available_time and self.next_model < self.N:
                start_inference = time.perf_counter()
                model = self.models[self.next_model]
                ret = None
                if not self.errors[self.next_model]: 
                    try:
                        ret = model(x)
                    except ValueError:
                        pass

                    if not self.errors[self.next_model]:   
                        self.out_temp.append(ret)
                        self.used_model.append(self.next_model)

                torch.cuda.synchronize()
                end_inference = time.perf_counter()
                available_time -= (end_inference - start_inference)*1000

                # TODO if you have more than 3 models
                # write a function to find the next model here based on self.used_models
                # next_model = get_the_next_model(self.used_models)
                # or if there are no other option, break
                self.next_model = self.next_model + 1 # for now
                break
            else:
                break
        
        out = torch.mean(torch.stack(self.out_temp), dim=0)
        return out, available_time

    def forward(self, x=None, limit_model=2, available_time=1e9):
        empty = True
        self._reset_models()

        for i in range(min(self.N, limit_model)):
            model = self.models[i]
            ret = None
            if not self.errors[i] and available_time > self.wec_time: 
                
                start_inference = time.perf_counter()
                try:
                    ret = model(x)
                except ValueError:
                    pass
                torch.cuda.synchronize()
                end_inference = time.perf_counter()
                available_time -= (end_inference - start_inference)*1000

                if not self.errors[i]:   
                    self.out_temp.append(ret)
                    self.used_model.append(i)
                    empty = False
                if self.debug:
                    print(i,"=> inf_time",(end_inference - start_inference)*1000, "|", available_time)

        if empty: 
            out = torch.rand((x.shape[0], self.output_class)).to(device)
        else:
            out = torch.mean(torch.stack(self.out_temp), dim=0)
        return out
    

    
class NetDRO(nn.Module):
    
    def __init__(self, model):
        
        super(NetDRO, self).__init__()
        self.model = copy.deepcopy(model)
        self.backup1 = nn.ModuleDict()
        self.backup2 = nn.ModuleDict()


        self.protected = False
        self.debug = False
        for name, module in self.model.named_modules():
            if len(list(module.children()))==0:
                module.__name__ = name.replace(".", "[dot]")
                with torch.no_grad():
                    self.backup1[module.__name__] = copy.deepcopy(module)
                    self.backup2[module.__name__] = copy.deepcopy(module)
                module.register_forward_pre_hook(self.tmr())

    def tmr(self):
        def hook(module, inp):
            with torch.no_grad():
                if self.protected:
                    module_backup_1 = self.backup1[module.__name__]
                    module_backup_2 = self.backup2[module.__name__]
                    for param_0, param_1, param_2 in zip(module.parameters(), module_backup_1.parameters(), module_backup_2.parameters()):
                        flag = 0
                        if torch.allclose(param_0, param_1):
                            if not torch.allclose(param_1, param_2):
                                flag += 2
                        else:
                            flag += 1
                            if not torch.allclose(param_1, param_2):
                                flag += 2
                                if not torch.allclose(param_0, param_2):
                                    flag += 4
                        if flag==1:
                            param_0.copy_(param_1)
                        elif flag==3:
                            param_1.copy_(param_0)
                        elif flag==2:
                            param_2.copy_(param_0)
        return hook

    def forward(self, x):
        out = self.model(x)
        return out

class NetTMR(nn.Module):
    
    def __init__(self, model):
        
        super(NetTMR, self).__init__()
        self.model = copy.deepcopy(model)
        self.backup1 = nn.ModuleDict()
        self.backup2 = nn.ModuleDict()

        self.protected = False
        self.debug = False
        for name, module in self.model.named_modules():
            if len(list(module.children()))==0:
                module.__name__ = name.replace(".", "[dot]")
                with torch.no_grad():
                    self.backup1[module.__name__] = copy.deepcopy(module)
                    self.backup2[module.__name__] = copy.deepcopy(module)
                module.register_forward_hook(self.tmrout())

    def tmrout(self):
        def hook(module, inp, out):
            with torch.no_grad():
                if self.protected:
                    module_1 = self.backup1[module.__name__].eval()
                    module_2 = self.backup2[module.__name__].eval()
                    
                    out_backup_1 = module_1(inp[0])
                    out_backup_2 = module_2(inp[0])
                    out_correct = None # correct output
                    flag = 0 # check error
                    if torch.allclose(out, out_backup_1, equal_nan=True):
                        if not torch.allclose(out_backup_1, out_backup_2, equal_nan=True):
                            flag += 2
                            out_correct = out
                    else:
                        flag += 1
                        out_correct = out_backup_1[:]
                        if not torch.allclose(out_backup_1, out_backup_2, equal_nan=True):
                            flag += 2
                            out_correct = out
                            if not torch.allclose(out, out_backup_2, equal_nan=True):
                                flag += 4  

                
                    for param_0, param_1, param_2 in zip(module.parameters(), self.backup1[module.__name__].parameters(), self.backup2[module.__name__].parameters()):
                        if flag==1:
                            param_0.copy_(param_1)
                        elif flag==3:
                            param_1.copy_(param_0)
                        elif flag==2:
                            param_2.copy_(param_0)
                    return out_correct
                                
        return hook

    def forward(self, x):
        out = self.model(x)
        return out

class NetDMR(nn.Module):
    
    def __init__(self, model, path, sum):
        
        super(NetDMR, self).__init__()
        self.model = copy.deepcopy(model)
        self.backup_path = path
        self.sums = sum
        self.protected = False
        self.debug = False

        reg_counter = 0
        for name, module in self.model.named_modules():
            if len(list(module.children()))==0:
                module.__name__ = name.replace(".", "[dot]")
                module.__idx__ = reg_counter
                reg_counter += 1
                module.register_forward_pre_hook(self.dmr())

        for name, p in self.model.named_parameters():
            torch.save(p, os.path.join("dmr_backup", self.backup_path, str(name)+".pt"))
            

    def dmr(self):
        def hook(module, inp):
            with torch.no_grad():
                if self.protected:
                    sum_new = sum_module(module)

                    for i, (n, p_0) in enumerate(module.named_parameters()):
                        sum_0 = sum_new[i]
                        sum_1 = self.sums[module.__idx__][i]
                        fault_flag = False
                        if not torch.isclose(sum_0, sum_1, rtol=1e-08):
                            fault_flag = True

                        if fault_flag:
                            module_name = module.__name__.replace("[dot]", ".") + "." + n + ".pt"
                            loaded_module = torch.load(os.path.join("dmr_backup", self.backup_path, module_name))
                            p_0.copy_(loaded_module)
        return hook

    def forward(self, x):
        out = self.model(x)
        return out


class NetETF(nn.Module):
    
    def __init__(self, models):
        
        super(NetETF, self).__init__()
        self.models = copy.deepcopy(models)
        self.clean_models = copy.deepcopy(models)
        self.N = len(self.models)
        self.used_model = [1,1,1]
        self.errors = None
        self.protected = False
        self.debug = False
        self.counter = [0]*self.N

    def _reset_models(self):
        pass

    def _clear_errors(self):
        self.models = copy.deepcopy(self.clean_models)
        self.counter = [0]*self.N

    def forward(self, x, *args, **kwargs):
        outs = []
        out = []
        for i in range(self.N):
            outs.append(self.models[i](x))

        # vote and return the outs with highest vote for each data in batch
        for i in range(x.shape[0]):
            pred_0 = torch.argmax(outs[0][i])
            pred_1 = torch.argmax(outs[1][i])
            pred_2 = torch.argmax(outs[2][i])

            if pred_0 == pred_1:
                out.append(outs[1][i])
            elif pred_0 == pred_2:
                out.append(outs[2][i])
            elif pred_1 == pred_2:
                out.append(outs[2][i])
            else:
                # return the average
                out.append((outs[0][i] + outs[1][i] + outs[2][i])/3)

        return torch.stack(out)