import tensorflow       as tf
from   tensorflow       import keras
from   tensorflow.keras import layers
import numpy            as np
import pandas           as pd
from   PIL              import Image, ImageTk, ImageDraw, ImageFont
import tkinter          as tk
from   tkinter          import filedialog, messagebox, Label, Frame
#!pip install tabulate
from   tabulate         import tabulate
from   collections      import Counter
from   fpdf             import FPDF
from   prettytable      import PrettyTable
import os
import webbrowser
#import keras
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re

from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, Add, Conv2D, Activation, Lambda, GlobalMaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
import tensorflow.keras.backend as K

from tensorflow_addons.optimizers import AdamW
# Initialize the AdamW optimizer
optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

# Update category_colors mapping
category_colors = {
    'Fat': 'yellow',
    'Stroma': 'green',
    'Tumor': 'red'
}

# Update color_values with the new colors
color_values = {
    'red': (255, 0, 0),          # Red for Tumor
    'yellow': (255, 255, 153),   # Yellow for Fat
    'green': (0, 204, 102)      # Green for Stroma
}

# Update color_dict with the new colors
color_dict = {
    'red': (255, 0, 0),          # Red
    'yellow': (255, 255, 0),     # Yellow
    'green': (0, 128, 0)        # Green
}

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, embed_dim, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

def se_block(input_tensor, reduction=16):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]

    # Channel-wise attention (SE Block)
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // reduction, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = Multiply()([input_tensor, se])

    # Spatial attention
    spatial = Conv2D(filters // reduction, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(input_tensor)
    spatial = BatchNormalization()(spatial)
    spatial = Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal', use_bias=False)(spatial)

    # Apply spatial attention
    spatial = Multiply()([input_tensor, spatial])

    # Combine channel-wise and spatial attentions
    x = Add()([input_tensor, se, spatial])
    return x

# Convolutional Block Attention Module (CBAM)
def cbam_block(input_tensor, reduction=16):
    channel = input_tensor.shape[-1]
    
    # Channel attention
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    max_pool = Reshape((1, 1, channel))(max_pool)
    dense1 = Dense(channel // reduction, activation='relu', kernel_initializer='he_normal', use_bias=False)
    dense2 = Dense(channel, kernel_initializer='he_normal', use_bias=False)
    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))
    channel_attention = layers.Add()([avg_out, max_out])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    x = Multiply()([input_tensor, channel_attention])
    
    # Spatial attention
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = Conv2D(1, (7, 7),dilation_rate= 2, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    x = Multiply()([x, spatial_attention])
    
    return x

# Create a Tkinter window
root = tk.Tk()
root.title("Histo-Cell CAD (1.0.0)")

# Function to load the addresses of the patient's tiles and description file
def load_addresses():
    tiles_folder = filedialog.askdirectory(title="Select Tiles Folder")
    description_file = filedialog.askopenfilename(title="Select Description File (e.g. tiles.csv)")
    return tiles_folder, description_file


'''# Load the pre-trained model
def load_model():
    #custom_objects = {'TransformerBlock': TransformerBlock}
    #model_path = filedialog.askopenfilename(title="Select Pre-trained Model")
    model_path = 'best_model/T_Neur_Train_Val_Test.hdf5'
    custom_objects = {'TransformerBlock': TransformerBlock}
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile = False)
    # Load the weights separately
    model.load_weights(model_path)
    # Optionally compile the model
    TN = keras.metrics.TrueNegatives()
    TP = keras.metrics.TruePositives()
    FN = keras.metrics.FalseNegatives()
    FP = keras.metrics.FalsePositives()
    SEN = keras.metrics.Recall()
    AUC = keras.metrics.AUC()
    SPE = tf.keras.metrics.SpecificityAtSensitivity(0.5)
    SENS= tf.keras.metrics.SensitivityAtSpecificity(0.5)
    model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy', TN, TP, FN, FP, SEN, AUC, SPE, SENS])
    return model'''

# Define the weighted loss function
def weighted_wights():
    # Define the number of patients for each class in the training, validation, and test sets
    class_patients = {
        'Yellow': [106, 13, 26],  # Fat
        'Green': [110, 14, 26],   # Stroma
        'Red': [80, 11, 23]      # Tumor
     }

    # Calculate the frequency of each class in each set
    class_frequencies = {}
    for class_name, patients in class_patients.items():
        class_frequencies[class_name] = sum(patients)

    # Calculate the sum of frequencies for all classes
    total_frequency = sum(class_frequencies.values())
    # Normalize frequencies to obtain class weights
    class_weights = {class_name: frequency / total_frequency for class_name, frequency in class_frequencies.items()}
    # Print the class weights
    print("Class Weights:")
    for class_name, weight in class_weights.items():
        print(f"{class_name}: {weight:.4f}")
weighted_wights()

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute cross entropy loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Compute focal loss
        focal_loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed

# Define weighted focal loss
def weighted_focal_loss(gamma=2.0):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        prob_true = tf.reduce_sum(y_true * y_pred, axis=-1)
        weight = tf.reduce_sum(y_true, axis=-1)
        focal_loss = weight * tf.pow((1.0 - prob_true), gamma) * cross_entropy
        return tf.reduce_mean(focal_loss)
    return focal_loss

def load_model():
    # Specify the path to the pre-trained model
    model_path = 'best_model/L_ViT_fold_cv3_1.hdf5'
    
    # Define custom objects if any are used in the model
    custom_objects = {'TransformerBlock': TransformerBlock, 'cbam_block':cbam_block, 'se_block': se_block}
    
    '''# Load the pre-trained model, excluding compilation
    model = keras.models.load_model(model_path, custom_objects = custom_objects, compile = True)
    model.load_weights(model_path)'''


    model = keras.models.load_model(model_path, custom_objects = custom_objects, compile = False)
    # Optionally, load the weights separately (not necessary if already loaded with load_model)
    # model.load_weights(model_path)
    
    # Define metrics for evaluation
    TN = keras.metrics.TrueNegatives()
    TP = keras.metrics.TruePositives()
    FN = keras.metrics.FalseNegatives()
    FP = keras.metrics.FalsePositives()
    SEN = keras.metrics.Recall()
    AUC = keras.metrics.AUC()
    SPE = tf.keras.metrics.SpecificityAtSensitivity(0.5)
    SENS = tf.keras.metrics.SensitivityAtSpecificity(0.5)
    
    # Compile the model with specified optimizer, loss function, and metrics
    #opt = tf.keras.optimizers.AdamW(learning_rate=0.001)  # You can specify your optimizer here
    model.compile(optimizer=optimizer,
                  loss= weighted_focal_loss(),
                  metrics=['accuracy', TN, TP, FN, FP, SEN, AUC, SPE, SENS])
    
    model.load_weights(model_path)
    
    return model


def combine_images_horizontally_pil(roi_image, predicted_image, predicted_image_with_transparency):
    # Open two image files
    image1 = roi_image
    image2 = predicted_image
    image3 = predicted_image_with_transparency
    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size
    width3, height3 = image3.size
    # Calculate the total width and height for the combined image
    total_width = width1 + width2 + width3
    max_height = max(height1, height2)
    # Create a blank image with the calculated size
    combined_image = Image.new('RGB', (total_width, max_height), color='white')
    # Paste the first image onto the combined image
    combined_image.paste(image1, (0, 0))
    # Paste the second image next to the first one
    combined_image.paste(image2, (width1, 0))
    # Paste the third image next to the first one
    combined_image.paste(image3, (width1 + width2, 0))
    
    return combined_image


def generate_pretty_table(patient_results):
    headers = ["Priority", "Cell Type", "IOU"]
    table_data = []
    for row in patient_results:
        # Extract primary cancer type by removing text within parentheses
        primary_cancer = re.sub(r'\([^)]*\)', '', row[1]).strip()
        table_data.append([row[0], primary_cancer, row[2]])

    return tabulate(table_data, headers=headers, tablefmt="fancy_grid")


def display_table_in_gui(results, roi_image, predicted_image, predicted_image_with_transparency, canvas, frame_width, frame_height):

    # Create a new Tkinter window each time
    table_root = tk.Toplevel(canvas)
    table_root.title("Diagnostic Results Generated by Histo-Cell CAD")

    # Function to handle the "Zoom" option
    def zoom_action():
        cih = combine_images_horizontally_pil(roi_image, predicted_image, predicted_image_with_transparency)
        cih.show()

    
    def parse_color(color_str):
        # Dictionary mapping named colors to their RGB values
        color_dict = {
            'red': (255, 0, 0),          # Red
            'yellow': (255, 255, 0),     # Yellow
            'green': (0, 128, 0)        # Green
            }
        # Check if color_str is a named color
        if color_str in color_dict:
            return color_dict[color_str]
        # If color_str is not a named color, try parsing it as a hex color
        color_str = color_str.lstrip('#')
        return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))


    def print_action():
        # Use file dialog to get the save location
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            # Create a PDF document with landscape orientation and A4 size
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()

            # Set font to Times New Roman and size 12 for the table heading
            pdf.set_font("Times", size=12, style='B')

            # Add the table heading to the PDF
            pdf.cell(277, 10, "Diagnostic Results Generated by Histo-Cell CAD", ln=True, align='C')
            pdf.cell(40, 10, "Priority", 1, 0, 'C')
            pdf.cell(120, 10, "Cancer", 1, 0, 'C')
            pdf.cell(117, 10, "Diagnosing Probability", 1, 1, 'C')

            # Set font to Times New Roman and size 10 for the table content
            pdf.set_font("Times", size=10)
            
            # Add the table rows to the PDF
            for row in results:
                row[1] = re.sub(r'\([^)]*\)', '', row[1]).strip()
                # Check if the color name exists in the category_colors dictionary
                if str(row[1]) in category_colors:
                    print(str(row[1]))
                    # Retrieve the corresponding color name
                    color_name = category_colors[str(row[1])]
                    
                    # Check if the retrieved color name exists in the color_values dictionary
                    if color_name in color_values:
                        # Get the RGB values from the color_values dictionary
                        color_rgb = color_values[color_name]
                        #color_rgb = parse_color(color_name)


                        # Set the PDF fill color using the obtained RGB values
                        pdf.set_fill_color(*color_rgb)  # Unpack RGB values

                # Set the background color
                #pdf.set_fill_color(255, 255, 0)  # Yellow color, change RGB values as needed
                # Draw a rectangle to serve as the background
                pdf.rect(pdf.get_x(), pdf.get_y(), 40, 10, 'F')

                # Write the cell content
                pdf.set_text_color(0, 0, 0)  # Black color for text
                pdf.cell(40, 10, str(row[0]), 1, 0, 'C')  # Use 0 for border argument to avoid drawing cell borders

                # Extract primary cancer type by removing text within parentheses
                #pdf.cell(40, 10, str(row[0]), 1, 0, 'C')
                
                pdf.cell(120, 10, str(row[1]), 1, 0, 'L')
                pdf.cell(117, 10, str(row[2]), 1, 1, 'C')

            # Save the images to temporary files
            roi_temp_path = "temp_roi.png"
            predicted_temp_path = "temp_predicted.png"
            transparency_temp_path = "temp_transparency.png"
            roi_image.save(roi_temp_path)
            predicted_image.save(predicted_temp_path)
            predicted_image_with_transparency.save(transparency_temp_path)

            # Add the images to the PDF
            max_width = pdf.w - 20
            image_height = max_width // 3

            pdf.image(roi_temp_path, x=10, y=pdf.get_y() + 20, w=max_width // 3)
            pdf.image(predicted_temp_path, x=10 + max_width // 3, y=pdf.get_y() + 20, w=max_width // 3)
            pdf.image(transparency_temp_path, x=10 + 2 * (max_width // 3), y=pdf.get_y() + 20, w=max_width // 3)


            # Add titles for the images
            pdf.set_font("Times", size=10, style='B')
            pdf.cell(max_width // 3, 10, "Reconstructed WSI", ln=0, align='C')
            pdf.cell(max_width // 3, 10, "Segmented WSI", ln=0, align='C')
            pdf.cell(max_width // 3, 10, "Probability-based segmented WSI", ln=1, align='C')


        # Remove the temporary image files
        os.remove(roi_temp_path)
        os.remove(predicted_temp_path)
        os.remove(transparency_temp_path)

        # Output the PDF document
        pdf.output(file_path)

    # Create a menu bar
    menu_bar = tk.Menu(table_root)
    table_root.config(menu=menu_bar)

    # Create a "File" menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)

    # Add "Zoom" option to the menu
    file_menu.add_command(label="Zoom", command=zoom_action)
    file_menu.add_separator()

    # Add "Print" option to the menu
    file_menu.add_command(label="Print", command=print_action)

    try:
        # Create a frame within the new Tkinter window
        table_frame = tk.Frame(table_root, padx=20, pady=20)
        table_frame.pack(side="top", fill='both')

        # Create a label to display the table
        table_label = tk.Label(table_frame, text=generate_pretty_table(results), justify='left', font=('Courier', 10))
        table_label.pack()

        # Create a frame to hold the legend
        legend_frame = tk.Frame(table_frame, padx=2, pady=2)
        legend_frame.pack(side='right', padx=2)

        category_colors = {
            'Fat': 'yellow',
            'Stroma': 'green',
            'Tumor': 'red'
            }

        # Create and display legend items
        for category, color in category_colors.items():
            item_frame = tk.Frame(legend_frame)
            item_frame.pack(anchor='w')
            color_label = tk.Label(item_frame, text='      ', bg=color, padx=2)
            color_label.pack(side='left')
            category_label = tk.Label(item_frame, text=category, padx=2)
            category_label.pack(side='left')

        # Resize images to match the frame dimensions
        roi_image = roi_image.resize((frame_width // 2, frame_height))
        predicted_image = predicted_image.resize((frame_width // 2, frame_height))
        predicted_image_with_transparency = predicted_image_with_transparency.resize((frame_width // 2, frame_height))

        # Convert PIL images to Tkinter PhotoImage
        roi_image_tk = ImageTk.PhotoImage(roi_image)
        predicted_image_tk = ImageTk.PhotoImage(predicted_image)
        predicted_image_with_transparency_tk = ImageTk.PhotoImage(predicted_image_with_transparency)

        # Create labels for displaying images with their titles
        image_frame1 = Frame(table_frame)
        image_frame1.pack(side='left', padx=2)
        Label(image_frame1, text="Reconstructed WSI").pack(side='bottom')
        Label(image_frame1, image=roi_image_tk).pack(side='top')

        image_frame2 = Frame(table_frame)
        image_frame2.pack(side='left', padx=2)
        Label(image_frame2, text="Segmented WSI").pack(side='bottom')
        Label(image_frame2, image=predicted_image_tk).pack(side='top')

        image_frame3 = Frame(table_frame)
        image_frame3.pack(side='left', padx=2)
        Label(image_frame3, text="Probability-based segmented WSI").pack(side='bottom')
        Label(image_frame3, image=predicted_image_with_transparency_tk).pack(side='top')

        # Keep references to PhotoImage objects to prevent garbage collection
        table_frame.roi_image_tk = roi_image_tk
        table_frame.predicted_image_tk = predicted_image_tk
        table_frame.predicted_image_with_transparency_tk = predicted_image_with_transparency_tk

    except Exception as e:
        print(f"Error: {e}")

    # Start the Tkinter main loop for the new window
    table_root.mainloop()


def classify_and_display():
    tiles_address, description_file = load_addresses()
    if not tiles_address or not description_file:
        return 
    
    # Load the tile information from CSV
    file_path = description_file

    # Set the folder path where the tiles are stored
    tiles_folder = tiles_address +'/' #folder_path

    # Load the tile information from the CSV file
    tile_info = pd.read_csv(file_path, delimiter=';')

    # Find the index of the last '/'
    last_slash_index = tiles_address.rfind('/')

    # Extract the substring including the last '/'
    db_address = tiles_address[:last_slash_index + 1]

    test_ds_reconstruct = tf.keras.utils.image_dataset_from_directory(
        db_address,
        labels     = 'inferred',
        label_mode ='categorical',
        #validation_split = 0.2,
        #subset = 'validation',
        shuffle    = False,
        image_size = (32, 32),
        batch_size = 512)
    
    # Calculate the number of images in the dataset
    num_images = sum(1 for _ in test_ds_reconstruct.unbatch())
    print("Number of images:", num_images)

    # Get the base name of the directory (folder) from the address
    folder_name = os.path.basename(os.path.dirname(tiles_address))
    print(folder_name)

    # Generate the folder names for each tile
    tile_folder_names = [f'{folder_name}_roi0_{str(i).zfill(8)}' for i in range(num_images)]
    T_Neur = load_model()
    # Call the function to display the message box after 5 seconds
    messagebox.showinfo("Notice!", "Please push the 'OK' button and wait for a while until the results are generated.")

    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print("GPU available:", gpu.name)
    else:
        print('GPU available: No')

    # Move the model to the GPU if available
    start_time = time.time()
    
    if gpus:
        gpu_index = None
        for i, gpu in enumerate(gpus):
            if 'NVIDIA' in gpu.name:
                gpu_index = i
                break
        if gpu_index is not None:
            with tf.device('/GPU:' + str(gpu_index)):
                pred = T_Neur.predict(test_ds_reconstruct)
        else:
            with tf.device('/GPU:0'):  # Use the first GPU as fallback
                pred = T_Neur.predict(test_ds_reconstruct)
    else:
        pred = T_Neur.predict(test_ds_reconstruct)

    elapsed_time = time.time() - start_time
    # Print the elapsed time in seconds
    print("Elapsed time for L_ViT prediction on all tiles: ", round(elapsed_time, 2), "seconds")

        
    '''gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print("GPU available:", gpu.name)
    else:
        print('GPU available: No')

    # Move the model to the GPU if availabl
    if gpus:
        gpu_names = [gpu.name for gpu in gpus]
        if 'NVIDIA' in gpu_names:
            gpu_index = gpu_names.index('NVIDIA')

        else:
            gpu_index = 0  # Use the first GPU if NVIDIA GPU is not found
        with tf.device('/GPU:' + str(gpu_index)):
            pred = T_Neur.predict(test_ds_reconstruct)
    else:
        pred = T_Neur.predict(test_ds_reconstruct)'''


    #pred = T_Neur.predict(test_ds_reconstruct)
    #print(pred)
    predictions = pred.argmax(axis=1)
    #print(predictions)

    cancer_categories = {
        0: 'Fat',
        1: 'Stroma',
        2: 'Tumor',
        }
    
    category_colors = {
            'Fat':       'yellow',
            'Stroma':   'green',
            'Tumor':     'red',
        }
    color_values = {
        'yellow': (255, 255, 153),     # Yellow
        'green': (0, 204, 102),        # Green
        'red': (255, 0, 0),       # Red
        }
    
    
    # Calculate the dimensions of the ROI
    min_x = tile_info['abs_position_x'].min()
    min_y = tile_info['abs_position_y'].min()
    max_x = tile_info['abs_position_x'].max() + tile_info['dimension_x'].max()
    max_y = tile_info['abs_position_y'].max() + tile_info['dimension_y'].max()
    print(tile_info.index)

    # Downsample the dimensions of the ROI image
    downsample_factor = 1  # Adjust as needed
    width = int((max_x - min_x) * downsample_factor)
    height = int((max_y - min_y) * downsample_factor)
    roi_image = Image.new('RGB', (width, height), color='white')
    
    '''# Iterate through each tile and paste it onto the reconstructed ROI image
    for index, row in tile_info.iterrows():
        if index % 100 == 0:
            print("Reconstracing. ROI tile number: ", index)
        folder_name = tile_folder_names[row['Tile_Number']]
        tile_path = tiles_folder + folder_name + '.png'
        tile = Image.open(tile_path)
        # Paste the modified tile onto the reconstructed ROI image
        roi_image.paste(tile, (int(row['abs_position_x'] - min_x), int(row['abs_position_y'] - min_y)))'''
    
    '''# Preload tiles into memory
    tiles = {}
    for index, row in tile_info.iterrows():
        folder_name = tile_folder_names[row['Tile_Number']]
        tile_path = os.path.join(tiles_folder, folder_name + '.png')
        tiles[index] = Image.open(tile_path)

    # Iterate through each tile and paste it onto the reconstructed ROI image
    for index, row in tile_info.iterrows():
        if index % 100 == 0:
            print("Reconstructing. ROI tile number:", index)
        tile = tiles[index]
        roi_image.paste(tile, (int(row['abs_position_x'] - min_x), int(row['abs_position_y'] - min_y)))

    # Close all opened tile images
    for tile in tiles.values():
        tile.close()'''
    

    # Function to load a tile
    def load_tile(tile_info, tiles_folder, tile_folder_names, index):
        folder_name = tile_folder_names[tile_info.iloc[index]['Tile_Number']]
        tile_path = os.path.join(tiles_folder, folder_name + '.png')
        return Image.open(tile_path)
    
    start_time = time.time()
    # Preload tiles into memory using multi-threading
    tiles = {}
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(load_tile, tile_info, tiles_folder, tile_folder_names, index): index for index in tile_info.index}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                tiles[index] = future.result()
            except Exception as exc:
                print(f'Exception occurred while loading tile {index}: {exc}')

    # Iterate through each tile and paste it onto the reconstructed ROI image
    for index, row in tile_info.iterrows():
        if index % 200 == 0:
            print("Image Reconstraction. Tile number:", index)
        tile = tiles[index]
        roi_image.paste(tile, (int(row['abs_position_x'] - min_x), int(row['abs_position_y'] - min_y)))

        # Close all opened tile images
    for tile in tiles.values():
        tile.close()

    # Calculate the target size in bytes (100 MB)
    target_size_bytes = 10 * 1024 * 1024  # 100 MB
    # Determine the scale factor to reduce the image size
    scale_factor = (target_size_bytes / len(roi_image.tobytes())) ** 0.5
    # Calculate the new dimensions based on the scale factor
    new_width = int(roi_image.width * scale_factor)
    new_height = int(roi_image.height * scale_factor)
    # Resize the image
    resized_roi_image = roi_image.resize((new_width, new_height))

    del roi_image
    # Save the resized image with compression to reduce file size
    resized_roi_image.save('path_to_save/roi_image.jpg', quality = 50)
    #resized_roi_image.show()
    elapsed_time = time.time() - start_time
    # Print the elapsed time in seconds
    print("Elapsed time for Image Reconstraction: ", round(elapsed_time, 2), "seconds")
    
    predicted_image = Image.new('RGB', (width, height), color='white')
    for index, category in enumerate(predictions):
        if index % 200 == 0:
            print("Class Prediction Heatmap is being drawn. Tile number: ", index)
        color = category_colors[cancer_categories[category]]  # Get color for the predicted category
        folder_name = tile_folder_names[index]
        tile_path = tiles_folder + folder_name + '.png'
        tile = Image.open(tile_path)
        # Use the color to create a tile representing the predicted category
        colored_tile = Image.new('RGB', tile.size, color=color)
        # Paste the colored tile onto the predicted image at the corresponding position
        predicted_image.paste(colored_tile, (int(tile_info.loc[index, 'abs_position_x'] - min_x), int(tile_info.loc[index, 'abs_position_y'] - min_y)))

    # Calculate the target size in bytes (100 MB)
    target_size_bytes = 10 * 1024 * 1024  # 100 MB
    # Determine the scale factor to reduce the image size
    scale_factor = (target_size_bytes / len(predicted_image.tobytes())) ** 0.5
    # Calculate the new dimensions based on the scale factor
    new_width = int(predicted_image.width * scale_factor)
    new_height = int(predicted_image.height * scale_factor)
    # Resize the image
    resized_predicted_image = predicted_image.resize((new_width, new_height))
    del predicted_image
    # Save the resized image with compression to reduce file size
    resized_predicted_image.save('path_to_save/predicted_image.jpg', quality = 100)
    #predicted_image.save('path_to_save/predicted_image.png')

    '''# Downsample the dimensions of the predicted image with transparency
    transparent_predicted_image = Image.new('RGBA', (width, height), color=(255, 255, 255, 255))
    start_time = time.time()
    for index, category in enumerate(predictions):
        if index % 100 == 0:
            print("Probability Map drawing. Tile number: ", index)
        color = category_colors[cancer_categories[category]]  # Get color for the predicted category
        # Calculate the maximum probability for the current tile
        max_probability = max(pred[index])
        
        # Get the RGB values for the color
        rgb_value = color_values[color]
        # Set color with transparency based on the maximum probability
        rgba_color = rgb_value + (int((max_probability) * 255),)
        rgba_color = rgba_color[:3] + (int((max_probability) * 255),)  # Add transparency value to RGB tuple

        folder_name = tile_folder_names[index]
        tile_path = tiles_folder + folder_name + '.png'
        tile = Image.open(tile_path)
        tile = tile.convert('RGBA')  # Convert to RGBA mode to handle transparency
        data = tile.getdata()
        # Replace black pixels with white
        new_data = []
        for item in data:
            # Check if the pixel is black or mostly black (R+G+B <= threshold)
            if sum(item[:3]) <= 50:  # You can adjust the threshold as needed
                new_data.append((255, 255, 255, 0))  # Set the pixel as transparent white
            else:
                new_data.append(item)
        
        tile.putdata(new_data)
        #tile = tile.convert('RGB')

        # Create a transparent RGBA image
        transparent_tile = Image.new('RGBA', tile.size, rgba_color)
        # Paste the transparent tile onto the predicted image at the corresponding position
        transparent_predicted_image.paste(transparent_tile, (int(tile_info.loc[index, 'abs_position_x'] - min_x), int(tile_info.loc[index, 'abs_position_y'] - min_y)), mask=transparent_tile)
        # Convert RGBA image to RGB
        #transparent_predicted_image = transparent_predicted_image.convert('RGB')

    #resized_transparent_predicted_image = resize_fun(transparent_predicted_image)
    # Calculate the target size in bytes (100 MB)
    target_size_bytes = 100 * 1024 * 1024  # 100 MB
    # Determine the scale factor to reduce the image size
    scale_factor = (target_size_bytes / len(transparent_predicted_image.tobytes())) ** 0.5
    # Calculate the new dimensions based on the scale factor
    new_width = int(transparent_predicted_image.width * scale_factor)
    new_height = int(transparent_predicted_image.height * scale_factor)
    # Resize the image
    resized_transparent_predicted_image = transparent_predicted_image.resize((new_width, new_height))

    del transparent_predicted_image
    # Save the resized image with compression to reduce file size
    resized_transparent_predicted_image  = resized_transparent_predicted_image.convert('RGB')
    resized_transparent_predicted_image.save('path_to_save/transparent_predicted_image.jpg', quality=90)'''
    #predicted_image.save('path_to_save/predicted_image.png')
    # Create a blank RGBA image to paste tiles onto

    # Downsample the dimensions of the predicted image with transparency
    transparent_predicted_image = Image.new('RGBA', (width, height), color=(255, 255, 255, 255))

    batch_size = 1024
    ind = 0
    start_time = time.time()
    for batch_start in range(0, len(predictions), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(predictions)))
        batch_images = []

        for index in batch_indices:
            ind = ind + 1
            if ind % 200 == 0:
                print("Probability Heatmap is being drawn. Tile number: ", ind)
            color = category_colors[cancer_categories[predictions[index]]]  # Get color for the predicted category
            max_probability = max(pred[index])

            rgb_value = color_values[color]
            rgba_color = rgb_value + (int(max_probability * 255),)  # Add transparency value to RGB tuple

            folder_name = tile_folder_names[index]
            tile_path = tiles_folder + folder_name + '.png'
            tile = Image.open(tile_path)#.convert('RGBA')
            data = np.array(tile)

            # Replace black pixels with white
            black_pixels = np.all(data[:, :, :3] <= 50, axis=2)
            data[black_pixels] = [255, 255, 255, 0]

            batch_images.append((data, rgba_color, (int(tile_info.loc[index, 'abs_position_x'] - min_x), int(tile_info.loc[index, 'abs_position_y'] - min_y))))

        for data, rgba_color, position in batch_images:
            tile = Image.fromarray(data)
            transparent_tile = Image.new('RGBA', tile.size, rgba_color)
            transparent_predicted_image.paste(transparent_tile, position, mask=transparent_tile)

        # Resize the image
        resized_transparent_predicted_image = transparent_predicted_image.resize((new_width, new_height))
        # Save the resized image with compression to reduce file size
        resized_transparent_predicted_image = resized_transparent_predicted_image.convert('RGB')
        resized_transparent_predicted_image.save('path_to_save/transparent_predicted_image.jpg', quality=50)
    elapsed_time = time.time() - start_time
    # Print the elapsed time in seconds
    print("Elapsed time for Visualizing the Probability Heatmap:", round(elapsed_time, 2), "seconds")

    # Assuming prob_score_test contains the predictions for all tiles for all patients
    # Reshape the predictions to group them per patient (assuming "num_images" tiles per patient)
    num_patients             = 1
    tiles_per_patient        = num_images
    prob_score_test_reshaped = pred.reshape(num_patients, tiles_per_patient, -1)
    predicted_labels         = []  # To store predicted labels from Majority Voting
    results                  = []  # Store results in a list of tuples

    # Perform majority voting for each patient
    for patient_idx in range(num_patients):
        patient_predictions = prob_score_test_reshaped[patient_idx]

        # Initialize counters for each patient
        prediction_counts = Counter()

        # Count occurrences of each cancer category for this patient
        for tile_predictions in patient_predictions:
            predicted_label = int(np.argmax(tile_predictions))  # Assuming predictions are one-hot encoded
            prediction_counts[predicted_label] += 1

        # Find the most frequent prediction
        primary_diagnosis_label, count = prediction_counts.most_common(1)[0]
        # Get the primary diagnosis category
        primary_diagnosis_category = cancer_categories[primary_diagnosis_label]
        # Get the probabilities for each category
        total_predictions = sum(prediction_counts.values())
        probabilities = {cancer_categories[label]: count / total_predictions for label, count in prediction_counts.items()}
        # Adjust probabilities to ensure they sum up to 100%
        total_prob = sum(probabilities.values())
        if total_prob != 1.0:
            for key in probabilities:
                probabilities[key] /= total_prob

        # Store results for this patient
        patient_results = []
        # Sort probabilities based on probability value
        #sorted_probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        sorted_probabilities = sorted(probabilities.items(), key=lambda x: (x[0] != "Non Tumor", x[1]), reverse=True)
        for idx, (category, probability) in enumerate(sorted_probabilities, start=1):
            priority_number = idx if category != "Non Tumor" else "-"
            if idx == primary_diagnosis_label + 1:
                # Primary diagnosis row
                patient_results.append([priority_number, f"{category} ({cancer_categories[primary_diagnosis_label]})", f"{probability * 100:.2f}%"])
            else:
                # Other categories
                patient_results.append([priority_number, category, f"{probability * 100:.2f}%"])
        # Append the results for this patient to the overall results
        results.append(patient_results)
        predicted_labels.append(primary_diagnosis_category)

    # Print the table for each patient
    for idx, patient_result in enumerate(results, start=1):
        print(f"Patient {idx}")
        print(tabulate(patient_result, headers=["Priority", "Cancer", "Diagnosing Probability"], tablefmt="pretty"))
        print("\n")

    # Call the function to display the table
    for idx, patient_result in enumerate(results, start=1):
        canvas = tk.Canvas(root, width=800, height=600)
        canvas.pack()
        # Set the dimensions based on the screen size
        screen_width  = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        width_factor  = 0.40#0.65
        height_factor = 0.40#0.75
        frame_width   = int(screen_width * width_factor)
        frame_height  = int(screen_height * height_factor)
        # Load the saved ROI image
        roi_image = Image.open('path_to_save/roi_image.jpg')
        predicted_image = Image.open('path_to_save/predicted_image.jpg')
        transparent_predicted_image = Image.open('path_to_save/transparent_predicted_image.jpg')
        display_table_in_gui(patient_result, roi_image, predicted_image, transparent_predicted_image, canvas, frame_width, frame_height)
    #messagebox.withdraw()

#This class is to add a background to the main frame
class Background_photo:
    def __init__(self, root):
        self.root = root
        # Load background image
        original_image = Image.open("GUI_Background.png")

        # Resize the image to a smaller size
        new_width     = 650  # Adjust as needed
        new_height    = int(original_image.height * (new_width / original_image.width))
        resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)
        self.background_photo = ImageTk.PhotoImage(resized_image)

        # Create a Canvas widget and add it to the window
        self.canvas = tk.Canvas(root, width = new_width, height = new_height)
        self.canvas.pack()

        # Draw the resized image on the Canvas
        self.canvas.create_image(0, 0, anchor = tk.NW, image = self.background_photo)

# Create an instance of YourApp class
app = Background_photo(root)


def open_web_application(url):
    webbrowser.open_new(url)


def on_exit():
    root.destroy()


tile_folder      = ""
description_file = ""

def open_file():
    tiles_folder, description_file = load_addresses()
    print(tiles_folder + "\n")
    print(description_file)
    print(f"The Images address and thier descriptopns were loaded. Press [Analysing and Diagnosing] buttom to analyse them")
    '''file_path = filedialog.askopenfilename(title="Open Data File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if file_path:
        print(f"Selected file: {file_path}")
        load_addresses()'''
    return tile_folder, description_file

# Set the dimensions based on the screen size
screen_width  = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
width_factor  = 0.33
height_factor = 0.66
width         = int(screen_width * width_factor)
height        = int(screen_height * height_factor)

# Calculate the x and y coordinates for centering the window
x = (screen_width - width) // 2
y = (screen_height - height) // 2
root.geometry(f"{width}x{height}+{x}+{y}")

# Create a menu bar
menu_bar         = tk.Menu(root)
root.config(menu = menu_bar)

####################################################################
# Create a "File" menu
File = tk.Menu(menu_bar, tearoff = 0)
menu_bar.add_cascade(label = "File", menu = File)
url_web = "https://tcnn-public-data-kzghiherap6vxmtnpzvikr.streamlit.app/"

# Add options to the "File" menu
File.add_command(label = "Diagnosis",       command = classify_and_display),                     File.add_separator()
File.add_command(label = "Web Application", command = lambda: open_web_application(url_web)),    File.add_separator()
File.add_command(label = "Exit",            command = on_exit)

# Create a frame to organize widgets
frame = tk.Frame(root, padx = 20, pady = 20)
frame.pack(expand = True, fill = 'both')
#####################################################################
# Create a "Help" menu
Help = tk.Menu(menu_bar, tearoff = 0)
menu_bar.add_cascade(label = "Help", menu = Help)

main_path   = ""
url_demo    = main_path + "/blob/main/Demo.gif"
url_doc     = main_path + "/blob/main/Documentations"
url_update  = main_path + "/blob/main/Updates"
url_licence = main_path + "/blob/main/LICENSE"
url_us      = main_path

# Add options to the "File" menu
Help.add_command(label = "Demo Video",       command = lambda: open_web_application(url_demo)),     Help.add_separator()
Help.add_command(label = "Documentations",   command = lambda: open_web_application(url_doc)),      Help.add_separator()
Help.add_command(label = "Check for Update", command = lambda: open_web_application(url_update)),   Help.add_separator()
Help.add_command(label = "Licence",          command = lambda: open_web_application(url_licence)),   Help.add_separator()
Help.add_command(label = "About Us",         command = lambda: open_web_application(url_us))

# Create a frame to organize widgets
frame = tk.Frame(root, padx = 20, pady = 20)
frame.pack(expand = True, fill = 'both')

# Start the Tkinter main loop
root.mainloop()