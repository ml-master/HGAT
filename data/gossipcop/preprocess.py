import json

# file_path = 'gossipcop_v3_origin.json'
# idx=1
# with (open(file_path, 'r',encoding="utf-8") as f,\
#     open('gossipcop.txt', 'w', encoding='utf-8') as outfile):
#     data = json.load(f)
#     for k,v in data.items():
#         text = ' '.join(v['text'].split("\n"))
#         formatted_line = f"{idx}\t{v['label']}\t{text}\n"
#         outfile.write(formatted_line)
#         idx+=1

file_path = 'gossipcop.txt'
with open(file_path, 'r',encoding='utf-8') as file:
    data = file.readlines()
    num = len(data)//6
    for i in range(6):
        with open('gossipcop'+str(i)+'.txt', 'w', encoding='utf-8') as outfile:
            outfile.write(''.join(data[i*num:(i+1)*num]))
    with open('gossipcop6.txt', 'w', encoding='utf-8') as outfile:
        outfile.write(''.join(data[6 * num:]))