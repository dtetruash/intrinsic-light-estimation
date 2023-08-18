from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

plt.imshow(np.random)

def echo(text):
    print(f"Hello, {text}!")


print("boo")
echo('World')


b = []
for i in tqdm(range(1000000000)):
    b.append(i)
