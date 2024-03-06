import imageio

FPS = 60

images = []
for i in range(1000):
    filename = f"cross_boundary_imgs/{i+1}.png"
    images.append(imageio.imread(filename))
imageio.mimsave(f"{FPS}.gif", images, fps=FPS)
