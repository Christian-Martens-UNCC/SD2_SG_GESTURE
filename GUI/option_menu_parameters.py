from option_menu_settings import *

# Functions


def return_main():
    # raise NotImplementedError
    """
    Place code that is used to return to the main screen controls here
    """


def return_options():
    # raise NotImplementedError
    """
    Place code that is used to return to the option's menu here
    """


def edit_setting(char, var_str):
    """
    :param char: The integer which you want to update the settings to (int)
    :param var_str: The setting that you want to update (str)
    :return: None
    """
    try:    # Checks to see if the input is valid
        option_menu_dic[var_str][str(char)]

    except:
        try:    # Checks if the option dictionary is a "storage" type
            option_menu_dic[var_str]["storage"]

        except:
            print("Not a valid input")

        else:
            new_str = var_str + " = " + str(
                char) + "\n"  # Creates a new string of the appropriate format for readlines()
            with open("option_menu_settings.py", "r+") as f:
                content = f.readlines()  # Stores the current settings

            with open("option_menu_settings.py", "w+") as f:  # Deletes current settings
                for line in content:  # Rewrites all old settings except the one which we want to change, which is updated to the new string
                    if line.find(var_str) != -1:
                        f.write(new_str)
                    else:
                        f.write(line)

            option_menu_dic[var_str]["stored_val"] = int(char)  # Updates the respective dictionary

    else:   # Runs this code if input is valid
        new_str = var_str + " = " + str(char) + "\n"    # Creates a new string of the appropriate format for readlines()
        with open("option_menu_settings.py", "r+") as f:
            content = f.readlines()     # Stores the current settings

        with open("option_menu_settings.py", "w+") as f:    # Deletes current settings
            for line in content:    # Rewrites all old settings except the one which we want to change, which is updated to the new string
                if line.find(var_str) != -1:
                    f.write(new_str)
                else:
                    f.write(line)

        option_menu_dic[var_str]["stored_val"] = int(char)     # Updates the respective dictionary


# Dictionaries

brightness_dic = {
    "stored_val": BRIGHTNESS,
    "1": 0.0,
    "2": 0.25,
    "3": 0.5,
    "4": 0.75,
    "5": 1.0,
    "6": 1.25,
    "7": 1.5,
    "8": 1.75,
    "9": 2.0,
    "11": return_options()
}

contrast_dic = {
    "stored_val": CONTRAST,
    "1": 0.0,
    "2": 0.25,
    "3": 0.5,
    "4": 0.75,
    "5": 1.0,
    "6": 1.25,
    "7": 1.5,
    "8": 1.75,
    "9": 2.0,
    "11": return_options()
}

camera_flip_dic = {
    "stored_val": CAMERA_FLIP,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "11": return_options()

}

red_shift_dic = {
    "stored_val": RED_SHIFT,
    "0": 0.0,
    "1": 0.2,
    "2": 0.4,
    "3": 0.6,
    "4": 0.8,
    "5": 1.0,
    "6": 1.2,
    "7": 1.4,
    "8": 1.6,
    "9": 1.8,
    "10": 2.0,
    "11": return_options()
}

green_shift_dic = {
    "stored_val": GREEN_SHIFT,
    "0": 0.0,
    "1": 0.2,
    "2": 0.4,
    "3": 0.6,
    "4": 0.8,
    "5": 1.0,
    "6": 1.2,
    "7": 1.4,
    "8": 1.6,
    "9": 1.8,
    "10": 2.0,
    "11": return_options()
}

blue_shift_dic = {
    "stored_val": BLUE_SHIFT,
    "0": 0.0,
    "1": 0.2,
    "2": 0.4,
    "3": 0.6,
    "4": 0.8,
    "5": 1.0,
    "6": 1.2,
    "7": 1.4,
    "8": 1.6,
    "9": 1.8,
    "10": 2.0,
    "11": return_options()
}

check_timer_dic = {
    "stored_val": TIMER,
    "storage": True
}

detect_con_dic = {
    "stored_val": DETECT_CON,
    "1": 0.1,
    "2": 0.2,
    "3": 0.3,
    "4": 0.4,
    "5": 0.5,
    "6": 0.6,
    "7": 0.7,
    "8": 0.8,
    "9": 0.9,
    "10": 1.0,
    "11": return_options()

}

tracking_con_dic = {
    "stored_val": TRACKING_CON,
    "1": 0.1,
    "2": 0.2,
    "3": 0.3,
    "4": 0.4,
    "5": 0.5,
    "6": 0.6,
    "7": 0.7,
    "8": 0.8,
    "9": 0.9,
    "10": 1.0,
    "11": return_options()

}

option_menu_dic = {
    "BRIGHTNESS": brightness_dic,
    "CONTRAST": contrast_dic,
    "CAMERA_FLIP": camera_flip_dic,
    "RED_SHIFT": red_shift_dic,
    "GREEN_SHIFT": green_shift_dic,
    "BLUE_SHIFT": blue_shift_dic,
    "TIMER": check_timer_dic,
    "DETECT_CON": detect_con_dic,
    "TRACKING_CON": tracking_con_dic,
    "MAIN": return_main()
}

option_menu_names = {
    "0": "BRIGHTNESS",
    "1": "CONTRAST",
    "2": "CAMERA_FLIP",
    "3": "RED_SHIFT",
    "4": "GREEN_SHIFT",
    "5": "BLUE_SHIFT",
    "6": "TIMER",
    "7": "DETECT_CON",
    "8": "TRACKING_CON",
    "11": "MAIN"
}
