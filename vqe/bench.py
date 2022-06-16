import json
import os
import sys

os.environ["OMP_NUMBER_THREADS"] = "3"

import bench_mindquantum
# import bench_paddle

def get_from_file(filename: str):
    path = os.getcwd()
    f = open(f"{path}/data/{filename}.txt", "r")
    s = f.read()
    data = json.loads(s)
    # print(data)
    f.close()
    return data


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        print("Please input name.")
    elif len(args) > 2:
        print("Please input only one name.")
    else:
        name = args[1]
        data = get_from_file(name)
        print(f"data={data}")

        iter_num = 50

        use_time = bench_mindquantum.bench(data, iter_num)
        # use_time = bench_paddle.bench(data, iter_num)