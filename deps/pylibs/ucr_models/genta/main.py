from tqdm import tqdm

from dataset_ucr import get_series
from first_div_second import main
import time

if __name__ == "__main__":
    error_file = open("error_pos.txt", "w+")
    correct_count = 0
    error_count = 0
    for i in tqdm(range(1, 250+1)):
        start_time = time.time()
        if i in [239,240,241]:
            error_count += 1
            continue
        print("\n")
        all_data, split_pos, anormal_range = get_series(i)
        pred_pos = main(all_data, split_pos)

        correct_range = (anormal_range[0]-100, anormal_range[1]+100)
        print(correct_range, pred_pos)

        ret = 1
        if pred_pos >= correct_range[0] and pred_pos <= correct_range[1]:
            ret = 1
        else:
            ret = -1
        if ret > 0: 
            correct_count += 1
        else: 
            error_count += 1
            error_file.write(f"{i}\n")
            error_file.flush()

        end_time = time.time()
        print (f"({i}) correct:", ret > 0, "================>correct_count:",correct_count, " error_count:", error_count, " time: %.1fs"%(end_time-start_time))
        
    error_file.close()