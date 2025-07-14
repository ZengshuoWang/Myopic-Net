from torch.optim import Adam

_opt_dict = {"Adam": Adam}


def get_opt(name):
    return _opt_dict[name]


def config_to_str(config):
    attrs = vars(config)  # 返回config配置参数中属性值的字典对象
    string_val = "Config: -----\n"
    string_val += "\n".join("%s: %s" % item for item in attrs.items())
    string_val += "\n----------"
    return string_val


def nice(dict):
    res = ""
    for k, v in dict.items():
        res += ("\t%s: %s\n" % (k, v))
    return res


def nice_2(a_list):
    res = ""
    for key in a_list:
        res += ("\t%s\n" % key)

    return res


def update_lr(optimiser, lr_mult=0.1):
    for param_group in optimiser.param_groups:
        param_group['lr'] *= lr_mult
    return optimiser
