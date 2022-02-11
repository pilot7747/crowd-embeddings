import sys
from benchmarks.benchmarks.benchmark import Benchmark
from benchmarks.utils import parse_args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    benchmark = None
    benchmark = Benchmark.setup(args)
    print('Running benchmark with args')
    print(args)
    print()
    test_results = benchmark.run()
    print('Done')
    print(test_results)

