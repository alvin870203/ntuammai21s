import os

def main():

    path = './data/train/C'
    for file_name in os.listdir(path):
        print(file_name)

if __name__ == "__main__":
    main()