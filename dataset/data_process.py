import pandas as pd
import pickle
csv = "./dataset1/mmsa.csv"
aligned_data = "./dataset1/aligned_50.pkl"
output_aligned = './dataset1/aligned_501.pkl'
no_aligned = "./dataset1/unaligned_50.pkl"
output_noalogiede = "./dataset1/no_align_501.pkl"

def main():
    labels = pd.read_csv(csv)
    data = pickle.load(open(aligned_data, 'rb'))
    no_data = pickle.load(open(no_aligned, 'rb'))
    for mode in ['train', 'valid', 'test']:
        list2 = labels[labels['mode'] == mode]['tgt'].tolist()   
        list2 = [eval(item) for item in list2]
        data[mode]['tgt'] = list2.copy()
        no_data[mode]['tgt'] = list2.copy()
    pickle.dump(data, open(output_aligned, 'wb'))
    pickle.dump(no_data, open(output_noalogiede, 'wb'))

if __name__ == '__main__':
    main()