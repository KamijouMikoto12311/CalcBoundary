import imageio

images = []
for i in range(1000):
    filename = f"imgs/{i+1}.png"
    images.append(imageio.imread(filename))
imageio.mimsave("plot_animation.gif", images, fps=5)
