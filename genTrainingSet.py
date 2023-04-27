file_r = open("corpus_才.txt","r",encoding="UTF-8")
file_w = open("datas_unclassified.csv","w",encoding="UTF-8")
rawdatas=[]
rawdatas=file_r.readlines()
file_r.close()
print("id","sentence","source","classification",sep=',',file=file_w)
for item in rawdatas:
    tmpLine={"id":"","sentence":"","source":""}
    tmpLine["id"]=item[0:item.index(':')].strip()
    tmpLine["source"]=item[item.index('【'):].strip()
    tmpLine["sentence"]=item[item.index(':')+1:item.index('【')].strip()
    # print("###")
    # print("--->"+tmpLine["id"])
    # print("--->"+tmpLine["sentence"])
    # print("--->"+tmpLine["source"])
    print(tmpLine["id"],tmpLine["sentence"],tmpLine["source"],"",sep=',',file=file_w)
file_w.close()
    
