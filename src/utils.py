from typing import List
class Utils:
    @staticmethod 
    def split_list_str(list_str:List[str]) -> List[list]:
        return [str.split() for str in list_str]
    def read_file_txt(file_path):
        texts=list()
        with open(file_path,'r',encoding='utf-8', errors='ignore') as f:
            for line in f:
                texts.append(line.strip())
        return texts