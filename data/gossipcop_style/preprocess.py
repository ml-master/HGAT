import json
import random

# file_path = 'gossipcop_v3-5_style_based_legitimate.json'
# idx=20000
# with (open(file_path, 'r',encoding="utf-8") as f,\
#     open('gossipcop_style_legitimate.txt', 'w', encoding='utf-8') as outfile):
#     data = json.load(f)
#     for k,v in data.items():
#         text = ' '.join(v['generated_text_t015'].split("\n"))
#         text = ' '.join(text.split("\t"))
#         formatted_line = f"{idx}\t{v['generated_label']}\t{text}\n"
#         outfile.write(formatted_line)
#         idx+=1
#
# file_path = 'gossipcop_style_legitimate.txt'
# with open(file_path, 'r',encoding='utf-8') as file:
#     data = file.readlines()
#     # 随机抽取3000条数据
#     random_data = random.sample(data, 3000)
#     # 将抽取的数据写入新的txt文件
#     with open('gossipcop_style_legitimate_3000.txt', 'w', encoding='utf-8') as new_file:
#         new_file.writelines(random_data)

res=[]
with (open('gossipcop_style2entity_fake_3000.txt', 'r',encoding="utf-8") as fake,\
      open('gossipcop_style2entity_legitimate_3000.txt', 'r',encoding="utf-8") as legi,\
    open('gossipcop_style2entity.txt', 'w', encoding='utf-8') as outfile):
    for line in fake:
        res.append(line.strip("\n")+"\n")
    for line in legi:
        res.append(line.strip("\n")+"\n")
    random.shuffle(res)
    outfile.writelines(res)


with open('gossipcop_style2entity.txt', 'r',encoding='utf-8') as file:
    data = file.readlines()
    print(data)