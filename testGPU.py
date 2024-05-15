import GPUtil
GPUs = GPUtil.getGPUs()
for gpu in GPUs:
    print(f"GPU: {gpu.name}, Load: {gpu.load*100}%, Memory Used: {gpu.memoryUsed}MB")
