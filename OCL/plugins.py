from avalanche.training.plugins import ReplayPlugin, LwFPlugin, CWRStarPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from OCL.agem import AGEMPlugin
from OCL.utils import extract_kwargs

def create_strategy(
        name: str,
        strategy_kwargs=None
    ):
    plugins = []

    if name == "er":
        specific_args = extract_kwargs(
            ["mem_size", "batch_size_mem"], strategy_kwargs
        )
        storage_policy = ClassBalancedBuffer(
            max_size=specific_args["mem_size"], adaptive_size=True
        )
        replay_plugin = ReplayPlugin(**specific_args, storage_policy=storage_policy)
        plugins.append(replay_plugin)
    
    elif name == "lwf":
        specific_args_lwf = extract_kwargs(
            ["alpha", "temperature"], strategy_kwargs
        )
        lwf_plugin = LwFPlugin(**specific_args_lwf)
        plugins.append(lwf_plugin)

    elif name == "er_lwf":
        specific_args_replay = extract_kwargs(
            ["mem_size", "batch_size_mem"], strategy_kwargs
        )
        specific_args_lwf = extract_kwargs(
            ["alpha", "temperature"], strategy_kwargs
        )
        storage_policy = ClassBalancedBuffer(
            max_size=specific_args_replay["mem_size"], adaptive_size=True
        )
        replay_plugin = ReplayPlugin(
            **specific_args_replay, storage_policy=storage_policy
        )
        lwf_plugin = LwFPlugin(**specific_args_lwf)
        plugins.append(replay_plugin)
        plugins.append(lwf_plugin)

    elif name == "agem":
        specific_args = extract_kwargs(
            ["mem_size", "sample_size"],
            strategy_kwargs,
        )
        agem_plugin = AGEMPlugin(**specific_args)
        plugins.append(agem_plugin)
    
    elif name == "cwr":
        specific_args = extract_kwargs(
            ["model", "freeze_remaining_model"],
            strategy_kwargs,
        )
        plugins.append(CWRStarPlugin(**specific_args))

    return plugins

