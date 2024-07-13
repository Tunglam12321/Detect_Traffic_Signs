import random

# Danh sách màu sắc
colors = [
    "red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "cyan", "magenta"
]
colors1 = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#800080", "#FFA500", "#FFC0CB", "#A52A2A", "#00FFFF", "#FF00FF"
]

# Function to assign colors to labels
def get_label_color(label, label_colors):
    if label not in label_colors:
        label_colors[label] = colors[len(label_colors) % len(colors)]
    return label_colors[label]

# Function to assign colors to labels
def get_label_color1(label, label_colors):
    if label not in label_colors:
        label_colors[label] = colors1[len(label_colors) % len(colors1)]
    return label_colors[label]

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def hex_to_bgr(hex_color):
    """Chuyển đổi màu hex thành BGR."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))