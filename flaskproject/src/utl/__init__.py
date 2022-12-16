@staticmethod
def spec(i):
    (lambda i: [print(f"--- 1.Shape ---\n{x.shape}\n"
                      f"--- 2.Features ---\n{x.columns}\n"
                      f"--- 3.Info ---\n{x.info}\n"
                      f"--- 4.Case Top1 ---\n{x.head(1)}\n"
                      f"--- 5.Case Bottom1 ---\n{x.tail(3)}\n"
                      f"--- 6.Describe ---\n{x.describe()}\n"
                      f"--- 7.Describe All ---\n{x.describe(include='all')}")
                for x in i])(i)