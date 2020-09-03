import os
def main():
    readme = open('/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/tienpv13/README.txt', 'r+')
    new_readme = open('/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/tienpv13/N_README.txt', 'w+')
    line = True
    while line:
        line = readme.readline().rstrip()
        new_line = '/media/data3/' + line.split('/', 3)[3]
        new_readme.write(new_line+'\n')
    readme.close()
    new_readme.close()
    return 0
if __name__ == "__main__":
    main()
