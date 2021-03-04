import os
id = '1'
files = os.listdir('./train/')
for i in files:
    print(i)
    with open('./train/'+i, "r") as f:
        lines = [line.rstrip('\n').replace('0', id, 1) for line in f]
        print(lines)
    f = open('./train/'+i, 'w')
    f.write('\n'.join(lines))
