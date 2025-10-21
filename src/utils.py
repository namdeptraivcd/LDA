from typing import List
class Utils:
    @staticmethod 
    def split_list_str(list_str:List[str]) -> List[list]:
        return [str.split() for str in list_str]