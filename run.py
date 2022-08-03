from mappo import Mappo
from buffer import Buffer

buffer = Buffer()
if __name__ == "__main__":
    ma = Mappo(21,5,2,2,2000,350,buffer)
    ma.run()