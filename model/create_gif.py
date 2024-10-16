# import os
# import imageio
# from PIL import Image

# def create_gif(base_output_dir, gif_name='training_progress.gif', duration=0.5, iterations=15):
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

#     # Convert images to the PIL format and set the duration for each frame
#     pil_images = [Image.fromarray(img) for img in images]
    
#     # Save the collected images as a GIF with specified duration
#     gif_path = os.path.join(base_output_dir, gif_name)
#     pil_images[0].save(
#         gif_path,
#         save_all=True,
#         append_images=pil_images[1:],
#         duration=duration * 1000,  # Convert duration from seconds to milliseconds
#         loop=0  # Loop forever
#     )

#     print(f"GIF saved at {gif_path}")

# if __name__ == "__main__":
#     # Define the base output directory where all iteration directories are located
#     base_output_dir = '/raven/u/dvoss/al_pmssmwithgp/model/plots'
    
#     # Optionally, you can customize the GIF name and duration between frames
#     gif_name = 'training_progress.gif'
#     duration = 0.5  # Duration in seconds between frames
#     iterations = 20  # Number of iterations you want to include in the GIF

#     # Create the GIF from the gp_plot.png files in each iteration directory
#     create_gif(base_output_dir, gif_name=gif_name, duration=duration, iterations=iterations)

import os
import imageio
from PIL import Image
from collections import defaultdict

def create_gifs_for_each_png(base_output_dir, gif_name_prefix='training_progress', duration=0.5, iterations=15):
    """
    Create GIFs from all .png files found in each iteration's output directory. 
    A separate GIF will be created for each unique .png filename across iterations.
    
    :param base_output_dir: The base directory where all iteration directories are stored.
    :param gif_name_prefix: Prefix for the output GIF files.
    :param duration: Duration between frames in seconds.
    :param iterations: Number of iterations to include in the GIF.
    """
    # Dictionary to hold lists of images by their filenames
    images_by_filename = defaultdict(list)
    
    # Collect .png files from each iteration directory, grouping them by filename
    for i in range(1, iterations + 1):
        iter_dir = os.path.join(base_output_dir, f'Iter{i}')
        
        if os.path.exists(iter_dir):
            # Find all PNG files in the iteration directory
            for file_name in os.listdir(iter_dir):
                if file_name.endswith('.png'):
                    plot_path = os.path.join(iter_dir, file_name)
                    images_by_filename[file_name].append(imageio.imread(plot_path))
                    print(f"Adding {plot_path} to the {file_name} GIF.")
        else:
            print(f"Warning: Directory {iter_dir} not found, skipping this iteration.")

    # For each unique filename, create a separate GIF
    for file_name, images in images_by_filename.items():
        if len(images) == 0:
            print(f"No images found for {file_name}, skipping GIF creation.")
            continue

        # Convert images to the PIL format and set the duration for each frame
        pil_images = [Image.fromarray(img) for img in images]
        
        # Define the GIF path based on the original file name
        gif_path = os.path.join(base_output_dir, f'{gif_name_prefix}_{file_name.replace(".png", "")}.gif')

        # Save the collected images as a GIF with specified duration
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
    base_output_dir = '/raven/u/dvoss/al_pmssmwithgp/model/plots/GIFs'
    
    # Optionally, you can customize the GIF name prefix, duration between frames, and the number of iterations
    gif_name_prefix = 'training_progress'  # This prefix will be added to each GIF name
    duration = 0.5  # Duration in seconds between frames
    iterations = 20  # Number of iterations you want to include in the GIF

    # Create separate GIFs for each .png file found in the iteration folders
    create_gifs_for_each_png(base_output_dir, gif_name_prefix=gif_name_prefix, duration=duration, iterations=iterations)
