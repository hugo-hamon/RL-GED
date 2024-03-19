from data_loader import *
import time

MAX_SIZE = 10

if __name__ == "__main__":
    time1 = time.time()
    data = generate_data(100, max_size=MAX_SIZE, verbose=True)
    print(f"Time: {time.time() - time1}, data length: {len(data)}")
