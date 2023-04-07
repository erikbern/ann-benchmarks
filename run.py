from multiprocessing import freeze_support

from ann_benchmarks.main import main

if __name__ == "__main__":
    freeze_support()
    main()
