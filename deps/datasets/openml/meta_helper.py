from pydantic import BaseModel


class MetaAutoCASH(BaseModel):
    """
    Auto-CASh 的特征定义
    the mete-features fine-tuned by DQN:
    mf0; mf2; mf4; mf6; mf7; mf9; mf13
    """
    id: str
    dataid: int
    data_name: str
    mf0: float = 0.0  # mf0: Class number in target attribute.
    mf2: float = 0.0  # mf2: Maximum proportion of single class in target attribute.
    mf4: float = 0.0  # mf4: Number of numeric attributes.
    mf6: float = 0.0  # mf6: Proportion of numeric attributes.
    mf7: float = 0.0  # mf7: Total number of attributes.
    mf9: float = 0.0  # mf9: Class number in category attribute with the least classes.
    mf13: float = 0.0  # mf13: Class number in category attribute with the most classes.

    def get_prompt_meta(self):
        """获取询问GPT的数据集的meta"""
        return (
            f"dataset name: {self.data_name}\n"
            f"Class number in target attribute:{self.mf0}\n"
            f"Maximum proportion of single class in target attribute:{self.mf2}\n"
            f"Number of numeric attributes:{self.mf4}\n"
            f"Proportion of numeric attributes:{self.mf6}\n"
            f"Total number of attributes:{self.mf7}\n"
            f"Class number in category attribute with the least classes:{self.mf9}\n"
            f"Class number in category attribute with the most classes:{self.mf13}\n"

        )
