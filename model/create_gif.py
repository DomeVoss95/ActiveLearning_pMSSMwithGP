# import os
# import imageio

# def create_gif(base_output_dir, gif_name='training_progress.gif', duration=3.5, iterations=20):
#     """
#     Create a GIF from the `gp_plot.png` files saved in each iteration's output directory.

#     :param base_output_dir: The base directory where all iteration directories are stored.
#     :param gif_name: Name of the output GIF file.
#     :param duration: Duration between frames in seconds.
#     :param iterations: Number of iterations to include in the GIF.
#     """
#     images = []
    
#     # Collect all gp_plot.png files from each iteration directory
#     for i in range(1, iterations + 1):
#         plot_path = os.path.join(base_output_dir, f'Iter{i}', 'gp_plot.png')
#         if os.path.exists(plot_path):
#             images.append(imageio.imread(plot_path))
#         else:
#             print(f"Warning: {plot_path} not found, skipping this iteration.")

#     # Save the collected images as a GIF
#     gif_path = os.path.join(base_output_dir, gif_name)
#     imageio.mimsave(gif_path, images, duration=duration)

#     print(f"GIF saved at {gif_path}")

# if __name__ == "__main__":
#     # Define the base output directory where all iteration directories are located
#     base_output_dir = '/raven/u/dvoss/al_pmssmwithgp/model/plots'
    
#     # Optionally, you can customize the GIF name and duration between frames
#     gif_name = 'training_progress.gif'
#     duration = 3.5  # Duration in seconds between frames
#     iterations = 20  # Number of iterations you want to include in the GIF

#     # Create the GIF from the gp_plot.png files in each iteration directory
#     create_gif(base_output_dir, gif_name=gif_name, duration=duration, iterations=iterations)

import os
import imageio
from PIL import Image

def create_gif(base_output_dir, gif_name='training_progress.gif', duration=0.5, iterations=15):
    """
    Create a GIF from the `gp_plot.png` files saved in each iteration's output directory.

    :param base_output_dir: The base directory where all iteration directories are stored.
    :param gif_name: Name of the output GIF file.
    :param duration: Duration between frames in seconds.
    :param iterations: Number of iterations to include in the GIF.
    """
    images = []
    
    # Collect all gp_plot.png files from each iteration directory
    for i in range(1, iterations + 1):
        plot_path = os.path.join(base_output_dir, f'Iter{i}', 'gp_plot.png')
        if os.path.exists(plot_path):
            images.append(imageio.imread(plot_path))
        else:
            print(f"Warning: {plot_path} not found, skipping this iteration.")

    # Convert images to the PIL format and set the duration for each frame
    pil_images = [Image.fromarray(img) for img in images]
    
    # Save the collected images as a GIF with specified duration
    gif_path = os.path.join(base_output_dir, gif_name)
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration * 1000,  # Convert duration from seconds to milliseconds
        loop=0  # Loop forever
    )

    print(f"GIF saved at {gif_path}")

if __name__ == "__main__":
    # Define the base output directory where all iteration directories are located
    base_output_dir = '/raven/u/dvoss/al_pmssmwithgp/model/plots'
    
    # Optionally, you can customize the GIF name and duration between frames
    gif_name = 'training_progress.gif'
    duration = 0.5  # Duration in seconds between frames
    iterations = 20  # Number of iterations you want to include in the GIF

    # Create the GIF from the gp_plot.png files in each iteration directory
    create_gif(base_output_dir, gif_name=gif_name, duration=duration, iterations=iterations)
