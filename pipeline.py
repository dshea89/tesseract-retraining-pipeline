import csv
import io
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import argparse
import tkinter as tk
from tkinter.filedialog import askopenfilename
from threading import Thread
from multiprocessing import freeze_support, set_start_method
from difflib import SequenceMatcher

import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import pytesseract
import tesserocr
from tesserocr import RIL, PSM, OEM, iterate_level
from PIL import Image, ImageDraw, ImageTk

from functools import partial
from time import time
from datetime import datetime
from tzlocal import get_localzone
import random

STANDARD_DPI = 500
SCALED_DOWN_DPI = 100
ORIGINAL_IMAGE = None
WORDS_DF = pd.DataFrame()
CHARS_DF = pd.DataFrame()

SCALE_FACTOR = STANDARD_DPI / SCALED_DOWN_DPI

OUTPUT_DIR = 'all_boxes'
OUTPUT_DIR_PATH = os.path.abspath('all_boxes')
LSTMF_FOLDER = 'lstmf'
RUNTIME_ID = str(int(time())) + str(random.randint(-sys.maxsize - 1, sys.maxsize))

TESSDATA_FOLDER = 'tessdata'

USE_TESSEROCR = False

USE_MSER_TO_FIND_LEFTOVER_REGIONS = True
LEFTOVER_OCR_REGION_PADDING = 10

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def convert_to_png(file_path, start_page=1, end_page=None, dpi=STANDARD_DPI, folder_directory=""):
    """
    Converts a file from PDF into a list of PNG images
    and returns the location of the image files
    pages is parsed as a list
    """
    start_page = str(start_page)
    if end_page:
        end_page = str(end_page)
        page = " -dFirstPage={} -dLastPage={}".format(start_page, end_page)
    else:
        page = " -dFirstPage={}".format(start_page)

    png_file_base, _ = os.path.splitext(os.path.basename(file_path))  # Separates file name from file extension
    png_file_path = os.path.dirname(os.path.abspath(file_path))

    def is_not_relative_path(path: str):
        if path.startswith("/") or path.startswith("\\"):  # UNIX
            return True
        if len(path) > 0:  # Windows
            if path[1] == ":":
                return True
        return False

    if is_not_relative_path(folder_directory):
        png_file_path = folder_directory
    elif folder_directory is not "":
        # When it's not a root directory, we assume its a sub folder and then join the two
        png_file_path = os.path.join(png_file_path, folder_directory)

    # png_file = os.path.abspath(file_path.replace(".pdf", ""))
    gs_process = ("gs", "gswin64c",)
    if os.name != "nt":
        gs = gs_process[0]
    else:
        gs = gs_process[1]

    output_file_gs = os.path.join(png_file_path, "{}-{}-%d.png".format(png_file_base, start_page))

    ghost_script_partial_command = "{} -dNOPAUSE -dBATCH -dTextAlphaBits=4 -sDEVICE=png16m -r{}{}".format(gs,
                                                                                                          str(dpi),
                                                                                                          page)
    output_script_partial_command = " -sOutputFile=\"{}\" \"{}\"".format(output_file_gs, file_path)
    png_command = ghost_script_partial_command + output_script_partial_command

    try:
        if end_page is not None:
            # Remove any existing output files of the same name since Ghostscript will throw an error otherwise
            for page_num in range(int(start_page), int(end_page) + 1):
                out_file = os.path.join(png_file_path, "{}-{}.png".format(png_file_base, page_num))
                if os.path.isfile(out_file):
                    os.remove(out_file)
    except:
        pass

    if os.name != 'nt':
        process = subprocess.Popen([png_command], shell=True)
        process.wait()
    else:
        os.system(png_command)

    # No guess work here, we use glob glob
    png_files = glob.glob(os.path.join(png_file_path, "{}-{}-*.png".format(png_file_base, start_page)))

    # Needs to sort png file alphabet#
    # inner function to get the numeric
    file_base_len = len(png_file_base) + 1

    rename_dict = {}

    def get_file_numeric(file_name):
        nonlocal rename_dict
        file_name = os.path.basename(file_name)

        numeric = file_name[file_base_len:-4]
        start, stop = map(int, numeric.split("-"))

        file_no = start + stop - 1

        rename_dict[numeric] = str(file_no)

        return file_no  # This will be used for sorting

    # Sorts it using numeric as key
    png_files = sorted(png_files, key=get_file_numeric)
    # Sorts rename dict
    rename_dict = sorted(rename_dict.items(), key=lambda kv: int(kv[1]))  # It is not a list of tuples

    thread_list = []
    renamed_png_files = []
    # Do not use async here and use threading because we want to keep the order of the png files in list.
    for fp, (old_sub, new_sub) in zip(png_files, rename_dict):
        new_fp = fp.replace(old_sub, new_sub)
        rename_thread = Thread(target=os.rename, args=(fp, new_fp))
        rename_thread.start()

        thread_list.append(rename_thread)
        renamed_png_files.append(new_fp)

    [thread.join() for thread in thread_list]

    return renamed_png_files


def similar(a, b, case_sensitive=True):
    if not case_sensitive:
        a = a.lower()
        b = b.lower()
    return SequenceMatcher(None, a, b).ratio()


def tesseract_cropped_region(img, xmin, ymin, xmax, ymax, starting_index):
    global USE_TESSEROCR

    words_df = pd.DataFrame()
    chars_df = pd.DataFrame()
    cropped_img = img.crop((xmin, ymin, xmax, ymax))

    if USE_TESSEROCR:
        # TODO
        pass
    else:
        try:
            # ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 1')
            ocr_output = pytesseract.image_to_data(cropped_img)
        except pytesseract.TesseractError:
            # ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 1 --oem 1')
            ocr_output = pytesseract.image_to_data(cropped_img, config='--oem 1')
        ocr_output = pd.read_csv(io.StringIO(ocr_output), sep="\t", quoting=csv.QUOTE_NONE, dtype={'text': object})
        ocr_output.dropna(subset=['text'], inplace=True)
        psm = '6'
        # psm = ''
        word_text = ""
        if len(ocr_output) <= 0:
            # Corner case in Tesseract where we need to treat the image as a single character
            try:
                ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 10')
            except pytesseract.TesseractError:
                ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 10 --oem 1')
            ocr_output = pd.read_csv(io.StringIO(ocr_output), sep="\t", quoting=csv.QUOTE_NONE, dtype={'text': object})
            ocr_output.dropna(subset=['text'], inplace=True)
            psm = '10'
        if len(ocr_output) > 0:
            # Build a DataFrame out of the extracted text
            ocr_output['text'] = ocr_output['text'].astype(str)
            ocr_output = ocr_output[ocr_output['text'].str.strip() != '']
            ocr_words = []
            ocr_chars = []
            cur_word_index = starting_index
            for ocr in ocr_output.itertuples():
                ocr_word = ocr.text
                ocr_xmin = int(ocr.left) + int(xmin)
                ocr_xmax = int(ocr.left) + int(ocr.width) + int(xmin)
                ocr_ymin = int(ocr.top) + int(ymin)
                ocr_ymax = int(ocr.top) + int(ocr.height) + int(ymin)
                new_ocr_word = add_word_to_df(ocr_word, ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax)

                # Get the characters that make up this word
                cropped_img = img.crop((ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax))
                if len(psm.strip()) == 0:
                    try:
                        ocr_box_output = pytesseract.image_to_boxes(cropped_img)
                    except pytesseract.TesseractError:
                        ocr_box_output = pytesseract.image_to_boxes(cropped_img, config='--oem 1')
                else:
                    try:
                        ocr_box_output = pytesseract.image_to_boxes(cropped_img, config='--psm ' + psm)
                    except pytesseract.TesseractError:
                        ocr_box_output = pytesseract.image_to_boxes(cropped_img, config='--psm ' + psm + ' --oem 1')
                ocr_box_output = pd.read_csv(io.StringIO(ocr_box_output),
                                             names=["symbol", "left", "bottom", "right", "top", "page"],
                                             delim_whitespace=True, quoting=csv.QUOTE_NONE, dtype={'symbol': object})
                if ocr_box_output.shape[0] > 0:
                    ocr_box_output['left'] += ocr_xmin
                    ocr_box_output['top'] = (cropped_img.size[1] - ocr_box_output['top']) + ocr_ymin
                    ocr_box_output['right'] += ocr_xmin
                    ocr_box_output['bottom'] = (cropped_img.size[1] - ocr_box_output['bottom']) + ocr_ymin
                    ocr_box_output['word'] = cur_word_index

                    # Because the MSER approach can potentially flag false positives, ensure that the characters
                    # make up the word, and only add if there is not a mismatch
                    if similar(ocr_word, "".join(ocr_box_output['symbol'].values.tolist())) > 0.75 and len(
                            re.findall(r'\w+', ocr_word)) > 0:
                        ocr_words.append(new_ocr_word)
                        ocr_chars.append(ocr_box_output)
                        cur_word_index += 1
            if len(ocr_words) > 0 and len(ocr_chars) > 0:
                words_df = words_df.append(pd.DataFrame(ocr_words), sort=False)
                words_df = words_df[words_df['value'].str.strip() != '']
                words_df.index += starting_index
                chars_df = chars_df.append(pd.concat(ocr_chars, ignore_index=True), sort=False)

    return words_df, chars_df


def identify_overlooked_regions_of_interest(words_df, img):
    # Depending on the page segmentation mode used, Tesseract may miss certain words. To reconicle this, apply the
    # maximally stable extremal regions (MSER) method to find where text is likely to be. If any of the identified
    # regions does not exist in the current cohort of identified words, then crop out that region and run it through
    # Tesseract for a closer examination.
    global STANDARD_DPI

    mser = cv2.MSER_create()

    # Convert to gray scale
    if type(img) == np.ndarray:
        img = Image.fromarray(np.uint8(img))
    img_data = np.asarray(img)
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

    vis = img_data.copy()

    # Detect regions in the grayscale image and group them to speed up comparisons
    regions, bboxes = mser.detectRegions(gray)
    bboxes = cv2.groupRectangles(list(bboxes), 1)[0]

    # Remove boxes that overlap with any of the identified Tesseract words
    filtered_bboxes = []
    for bbox in bboxes:
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        overlap_condition = ((words_df['left'] <= xmax) & (words_df['right'] >= xmin)) & \
                            ((words_df['top'] <= ymax) & (words_df['bottom'] >= ymin))
        if words_df[overlap_condition].shape[0] <= 0:
            # This bounding box has not yet been handled by Tesseract
            filtered_bboxes.append([xmin, ymin, xmax, ymax])
    if len(filtered_bboxes) == 0:
        # Nothing was found, likely due to being a page with no text
        return pd.DataFrame(), pd.DataFrame()

    # Cluster the remaining boxes together via DBSCAN
    roi_words_dfs = []
    roi_chars_dfs = []
    grouped_bboxes = []
    cur_word_index = max(words_df.index.tolist()) + 1
    clustering = DBSCAN(eps=STANDARD_DPI / 2.0, min_samples=2).fit(filtered_bboxes)
    # print(clustering.labels_)
    classes = np.unique(clustering.labels_).tolist()
    if -1 in classes:
        # -1 is considered noise by DBSCAN, so all -1 boxes must be processed separately
        noise_boxes = [filtered_bboxes[i] for i, label in enumerate(clustering.labels_) if label == -1]
        for noise_box in noise_boxes:
            xmin, ymin, xmax, ymax = noise_box
            overlap_condition = ((words_df['left'] <= xmax) & (words_df['right'] >= xmin)) & \
                                ((words_df['top'] <= ymax) & (words_df['bottom'] >= ymin))
            if ymax - ymin <= STANDARD_DPI / 100.0 or xmax - xmin <= STANDARD_DPI / 100.0 or \
                    words_df[overlap_condition].shape[0] > 0:
                continue
            roi_word_df, roi_char_df = tesseract_cropped_region(img, xmin - LEFTOVER_OCR_REGION_PADDING,
                                                                ymin - LEFTOVER_OCR_REGION_PADDING,
                                                                xmax + LEFTOVER_OCR_REGION_PADDING,
                                                                ymax + LEFTOVER_OCR_REGION_PADDING, cur_word_index)
            if roi_word_df.shape[0] > 0 and roi_char_df.shape[0] > 0:
                cur_word_index += roi_word_df.shape[0]
                roi_words_dfs.append(roi_word_df)
                roi_chars_dfs.append(roi_char_df)
            # cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        classes.remove(-1)
    for clazz in classes:
        class_boxes = [filtered_bboxes[i] for i, label in enumerate(clustering.labels_) if label == clazz]
        xmin = min(class_boxes, key=lambda x: x[0])[0]
        ymin = min(class_boxes, key=lambda x: x[1])[1]
        xmax = max(class_boxes, key=lambda x: x[2])[2]
        ymax = max(class_boxes, key=lambda x: x[3])[3]
        overlap_condition = ((words_df['left'] <= xmax) & (words_df['right'] >= xmin)) & \
                            ((words_df['top'] <= ymax) & (words_df['bottom'] >= ymin))
        if ymax - ymin <= STANDARD_DPI / 100.0 or xmax - xmin <= STANDARD_DPI / 100.0 or \
                words_df[overlap_condition].shape[0] > 0:
            continue
        roi_word_df, roi_char_df = tesseract_cropped_region(img, xmin - LEFTOVER_OCR_REGION_PADDING,
                                                            ymin - LEFTOVER_OCR_REGION_PADDING,
                                                            xmax + LEFTOVER_OCR_REGION_PADDING,
                                                            ymax + LEFTOVER_OCR_REGION_PADDING, cur_word_index)
        if roi_word_df.shape[0] > 0 and roi_char_df.shape[0] > 0:
            cur_word_index += roi_word_df.shape[0]
            roi_words_dfs.append(roi_word_df)
            roi_chars_dfs.append(roi_char_df)
        # cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # cv2.imwrite("test_output.png", vis)

    # Concatenate the list of DataFrame objects and return
    if len(roi_words_dfs) > 0 and len(roi_chars_dfs) > 0:
        roi_words_dfs = pd.concat(roi_words_dfs, ignore_index=True)
        roi_chars_dfs = pd.concat(roi_chars_dfs, ignore_index=True)
    else:
        roi_words_dfs = pd.DataFrame()
        roi_chars_dfs = pd.DataFrame()
    return roi_words_dfs, roi_chars_dfs


def add_word_to_df(ocr_word, ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax, flag=False):
    """
    Dict literal
    """
    new_word = dict(value=ocr_word,
                    xmin=ocr_xmin,
                    xmax=ocr_xmax,
                    ymin=ocr_ymin,
                    ymax=ocr_ymax,
                    top=ocr_ymin,
                    bottom=ocr_ymax,
                    left=ocr_xmin,
                    right=ocr_xmax,
                    topleft=[ocr_ymin, ocr_xmin],
                    topright=[ocr_ymin, ocr_xmax],
                    bottomleft=[ocr_ymax, ocr_xmin],
                    bottomright=[ocr_ymax, ocr_xmax],
                    width=ocr_xmax - ocr_xmin,
                    height=ocr_ymax - ocr_ymin,
                    flag=flag
                    )
    return new_word


def get_intersection_area(a, b):
    """
    Calculate the area of the intersection of two bounding boxes, returning 0 if they do not intersect.
    The shapes should be arrays of the form [min_x, min_y, max_x, max_y]
    """
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx >= 0 and dy >= 0 else 0


def draw_boxes(image, bounds, color):
    # Draw text borders around the image
    draw = ImageDraw.Draw(image)
    for bound in bounds.itertuples():
        draw.line((
            bound.xmin, bound.ymin,
            bound.xmax, bound.ymin,
            bound.xmax, bound.ymax,
            bound.xmin, bound.ymax,
            bound.xmin, bound.ymin), fill=color, width=5)
    return image


class TkDrawBorders(tk.Tk):
    def __init__(self, pdf_path, pg):
        global SCALE_FACTOR

        super().__init__()

        self.title("Tesseract Bound Identifier")

        self.tkgui = None

        self.NEW_BOXES = []
        self.IMG = None
        self.IMG_POINTER = 0
        self.IMG_INDEX = -1

        self.file_path = pdf_path
        self.page = pg

        # Convert the PDF page to PNG
        img_dir = os.path.dirname(self.file_path)
        os.chdir(img_dir)

        # Extract text and draw their borders
        self.words_df, self.chars_df, self.image = self.ocr_extraction()

        # Scale the image down to fit on the canvas
        maxsize = (max(self.image.width / SCALE_FACTOR, 1), max(self.image.height / SCALE_FACTOR, 1))
        self.image.thumbnail(maxsize, Image.ANTIALIAS)

        self.canvas = tk.Canvas(self)

        tk.Button(self, text='Undo', command=self.undo_box).pack(side=tk.TOP)
        tk.Button(self, text='Reset', command=self.clear_all_boxes).pack(side=tk.TOP)
        tk.Button(self, text='Finished', command=self.finish).pack(side=tk.TOP)

        # Load the modified image onto the canvas
        self.IMG = ImageTk.PhotoImage(self.image)
        self.canvas.delete(self.IMG_POINTER)
        self.IMG_POINTER = self.canvas.create_image(self.IMG.width() / 2, self.IMG.height() / 2, image=self.IMG)

        # Set up the mouse event bindings
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.canvas.tag_bind(self.IMG_POINTER, "<ButtonPress-1>", self.on_button_press)
        self.canvas.tag_bind(self.IMG_POINTER, "<B1-Motion>", self.on_move_press)
        self.canvas.tag_bind(self.IMG_POINTER, "<ButtonRelease-1>", self.on_button_release)

        # Build the display
        self.canvas.create_window(self.IMG.width() / 2, self.IMG.height() / 2)
        vbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=self.canvas.yview)
        hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.canvas.configure(xscrollcommand=hbar.set)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.canvas.xview)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.wm_geometry("{}x{}".format(self.IMG.width(), self.IMG.height() + 150))

    def ocr_extraction(self):
        global ORIGINAL_IMAGE
        global USE_TESSEROCR
        global USE_MSER_TO_FIND_LEFTOVER_REGIONS

        png_file_path = convert_to_png(self.file_path, start_page=self.page, end_page=self.page, dpi=STANDARD_DPI)[0]
        img = Image.open(png_file_path)
        ORIGINAL_IMAGE = img.copy()

        if USE_TESSEROCR:
            orig_width, orig_height = ORIGINAL_IMAGE.size
            words_df = pd.DataFrame()
            ocr_box_output = pd.DataFrame()
            ocr_words = []
            ocr_chars = []
            with tesserocr.PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.SPARSE_TEXT_OSD) as api:
                api.SetImage(img)
                api.Recognize()
                # The level on which extraction is to be done. Choices - BLOCK, TEXTLINE, WORD, SYMBOL
                level = RIL.WORD
                iterator = api.GetIterator()
                for r in iterate_level(iterator, level):
                    try:
                        # Get text and font attributes of the current patch
                        ocr_word = r.GetUTF8Text(level)
                        conf = r.Confidence(level)
                        ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax = r.BoundingBox(RIL.WORD)
                        new_ocr_word = add_word_to_df(ocr_word, ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax)
                        if len(ocr_word.strip()) == 0 or conf < 1.0 or (
                                len(ocr_word.strip()) <= 2 and new_ocr_word['width'] > (
                                orig_width / 2)) or ocr_xmin <= 0 or ocr_ymin <= 0 or ocr_xmax >= orig_width or \
                                ocr_ymax >= orig_height:
                            # This word is either empty or is a bug in Tesseract where an erroneous word has been
                            # identified
                            continue
                        ocr_words.append(new_ocr_word)
                        while not r.IsAtFinalElement(RIL.WORD, RIL.SYMBOL):
                            text = r.GetUTF8Text(RIL.SYMBOL)
                            left, top, right, bottom = r.BoundingBox(RIL.SYMBOL)
                            new_ocr_char = dict(symbol=text, left=left, bottom=bottom, right=right, top=top, page="0",
                                                word=len(ocr_words) - 1)
                            ocr_chars.append(new_ocr_char)
                            r.Next(RIL.SYMBOL)
                        text = r.GetUTF8Text(RIL.SYMBOL)
                        left, top, right, bottom = r.BoundingBox(RIL.SYMBOL)
                        new_ocr_char = dict(symbol=text, left=left, bottom=bottom, right=right, top=top, page="0",
                                            word=len(ocr_words) - 1)
                        ocr_chars.append(new_ocr_char)
                    except:
                        pass
            words_df = words_df.append(pd.DataFrame(ocr_words), sort=False)
            words_df = words_df.sort_values(['top', 'left'])
            ocr_box_output = ocr_box_output.append(pd.DataFrame(ocr_chars), sort=False)

            if words_df.shape[0] == 0 or ocr_box_output.shape[0] == 0:
                # There were no extracted words on this page to process, so return the default values
                return pd.DataFrame(), pd.DataFrame(), img
        else:
            try:
                # ocr_output = pytesseract.image_to_data(img, config='--psm 12')
                ocr_output = pytesseract.image_to_data(img)
            except pytesseract.TesseractError:
                # ocr_output = pytesseract.image_to_data(img, config='--psm 12 --oem 1')
                ocr_output = pytesseract.image_to_data(img, config='--oem 1')
            ocr_output = pd.read_csv(io.StringIO(ocr_output), sep="\t", quoting=csv.QUOTE_NONE, dtype={'text': object})
            ocr_output.dropna(subset=['text'], inplace=True)

            if len(ocr_output) == 0:
                # There were no extracted words on this page to process, so return the default values
                return pd.DataFrame(), pd.DataFrame(), img

            # Build a DataFrame out of the extracted text
            ocr_output['text'] = ocr_output['text'].astype(str)
            ocr_output = ocr_output[ocr_output['text'].str.strip() != '']
            words_df = pd.DataFrame()
            ocr_words = []
            for ocr in ocr_output.itertuples():
                ocr_word = ocr.text
                ocr_xmin = int(ocr.left)
                ocr_xmax = int(ocr.left) + int(ocr.width)
                ocr_ymin = int(ocr.top)
                ocr_ymax = int(ocr.top) + int(ocr.height)
                new_ocr_word = add_word_to_df(ocr_word, ocr_xmin, ocr_ymin, ocr_xmax, ocr_ymax)
                ocr_words.append(new_ocr_word)
            words_df = words_df.append(pd.DataFrame(ocr_words), sort=False)
            words_df = words_df[words_df['value'].str.strip() != '']

            # Drop any erroneous words due to Tesseract bugs and then sort
            orig_width, orig_height = ORIGINAL_IMAGE.size
            words_df = words_df[~((words_df['top'] <= 0) & (words_df['left'] <= 0) &
                                  (words_df['right'] >= orig_width) &
                                  (words_df['bottom'] >= orig_height))]
            words_df = words_df.sort_values(['top', 'left'])
            words_df.reset_index(inplace=True, drop=True)

            # Get bounding boxes for each character and match them with their appropriate word
            try:
                # ocr_box_output = pytesseract.image_to_boxes(img, config='--psm 12')
                ocr_box_output = pytesseract.image_to_boxes(img)
            except pytesseract.TesseractError:
                # ocr_box_output = pytesseract.image_to_boxes(img, config='--psm 12 --oem 1')
                ocr_box_output = pytesseract.image_to_boxes(img, config='--oem 1')
            ocr_box_output = pd.read_csv(io.StringIO(ocr_box_output),
                                         names=["symbol", "left", "bottom", "right", "top", "page"],
                                         delim_whitespace=True, quoting=csv.QUOTE_NONE, dtype={'symbol': object})
            ocr_box_output['top'] = img.size[1] - ocr_box_output['top']
            ocr_box_output['bottom'] = img.size[1] - ocr_box_output['bottom']
            ocr_char_to_word_link = []
            for ocr_box in ocr_box_output.itertuples():
                try:
                    word_df = words_df[((words_df['left'] <= ocr_box.right) & (words_df['right'] >= ocr_box.left)) &
                                       ((words_df['top'] <= ocr_box.bottom) & (words_df['bottom'] >= ocr_box.top))]
                    if len(word_df) == 1:
                        # Matched this character with one word
                        ocr_char_to_word_link.append(word_df.index[0])
                    elif len(word_df) > 1:
                        # More than one word matches this character, so select the word with the greatest overlap
                        max_intersection_area = -1
                        max_intersection_index = word_df.index[0]
                        for word in word_df.itertuples():
                            word_intersection_area = get_intersection_area([ocr_box.left, ocr_box.top, ocr_box.right,
                                                                            ocr_box.bottom],
                                                                           [word.left, word.top, word.right,
                                                                            word.bottom])
                            if word_intersection_area > max_intersection_area:
                                max_intersection_area = word_intersection_area
                                max_intersection_index = word.Index
                        ocr_char_to_word_link.append(max_intersection_index)
                    else:
                        ocr_char_to_word_link.append(None)
                except:
                    ocr_char_to_word_link.append(None)
            ocr_box_output['word'] = ocr_char_to_word_link
            ocr_box_output.dropna(inplace=True)

            # Add in any words and characters that were missed by Tesseract
            if USE_MSER_TO_FIND_LEFTOVER_REGIONS:
                overlooked_words_df, overlooked_chars_df = identify_overlooked_regions_of_interest(words_df, img)
                if overlooked_words_df.shape[0] > 0 and overlooked_chars_df.shape[0] > 0:
                    words_df = words_df.append(overlooked_words_df, sort=False)
                    words_df.reset_index(inplace=True, drop=True)
                    words_df = words_df.sort_values(['top', 'left'])
                    ocr_box_output = ocr_box_output.append(overlooked_chars_df, sort=False)
                    ocr_box_output.reset_index(inplace=True, drop=True)

        # Draw boxes around the extracted text
        draw_boxes(img, words_df, 'red')

        os.remove(png_file_path)
        return words_df, ocr_box_output, img

    def finish(self):
        # At this point, the user has indicated that they are finished providing text bounds
        global ORIGINAL_IMAGE
        global WORDS_DF
        global CHARS_DF

        # The image was scaled down for the purpose of presentation, so the coordinates must now
        # be scaled back to their target DPI
        new_box_coords = [[coord * SCALE_FACTOR for coord in self.canvas.coords(new_box)] for new_box in self.NEW_BOXES]

        # Add these bounds to the DataFrame object. Format for new_box_coords is (xmin, ymin, xmax, ymax)
        if len(new_box_coords) > 0:
            new_ocr_words = []
            new_ocr_chars = []
            for new_box in new_box_coords:
                # Attempt to guess the contents and character boxes of this new word
                self.chars_df['word'] += 1
                cropped_img = ORIGINAL_IMAGE.crop((new_box[0], new_box[1], new_box[2], new_box[3]))
                try:
                    ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 8')
                except pytesseract.TesseractError:
                    ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 8 --oem 1')
                ocr_output = pd.read_csv(io.StringIO(ocr_output), sep="\t", quoting=csv.QUOTE_NONE,
                                         dtype={'text': object})
                ocr_output.dropna(subset=['text'], inplace=True)
                psm = '12'
                word_text = ""
                if len(ocr_output) <= 0:
                    # Corner case in Tesseract where we need to treat the image as a single character
                    try:
                        ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 10')
                    except pytesseract.TesseractError:
                        ocr_output = pytesseract.image_to_data(cropped_img, config='--psm 10 --oem 1')
                    ocr_output = pd.read_csv(io.StringIO(ocr_output), sep="\t", quoting=csv.QUOTE_NONE,
                                             dtype={'text': object})
                    ocr_output.dropna(subset=['text'], inplace=True)
                    psm = '10'
                if len(ocr_output) > 0:
                    # The word has been found, so get the characters that make up the word
                    word_text = " ".join(list(map(str, ocr_output['text'].values))).strip()
                    try:
                        ocr_box_output = pytesseract.image_to_boxes(cropped_img, config='--psm ' + psm)
                    except pytesseract.TesseractError:
                        ocr_box_output = pytesseract.image_to_boxes(cropped_img, config='--psm ' + psm + ' --oem 1')
                    ocr_box_output = pd.read_csv(io.StringIO(ocr_box_output),
                                                 names=["symbol", "left", "bottom", "right", "top", "page"],
                                                 delim_whitespace=True, quoting=csv.QUOTE_NONE,
                                                 dtype={'symbol': object})
                    ocr_box_output['left'] += new_box[0]
                    ocr_box_output['top'] = (cropped_img.size[1] - ocr_box_output['top']) + new_box[1]
                    ocr_box_output['right'] += new_box[0]
                    ocr_box_output['bottom'] = (cropped_img.size[1] - ocr_box_output['bottom']) + new_box[1]
                    ocr_box_output['word'] = len(new_ocr_words)
                    new_ocr_chars.append(ocr_box_output)
                new_ocr_word = add_word_to_df(word_text, new_box[0], new_box[1], new_box[2], new_box[3], flag=True)
                new_ocr_words.append(new_ocr_word)
            self.words_df = pd.concat([pd.DataFrame(new_ocr_words), self.words_df], ignore_index=True)
            self.words_df.reset_index(inplace=True, drop=True)
            self.chars_df = pd.concat([pd.concat(new_ocr_chars), self.chars_df], ignore_index=True, sort=True)
            self.chars_df.reset_index(inplace=True, drop=True)

        WORDS_DF = self.words_df.copy()
        CHARS_DF = self.chars_df.copy()

        # Move onto the next phase
        self.destroy()
        self.tkgui = TkVerifyWords(words_df=WORDS_DF, chars_df=CHARS_DF)
        self.mainloop()

    def on_button_press(self, event):
        # Save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_move_press(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        # Expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        # Save this rectangle to the list of current custom rectangles
        self.NEW_BOXES.append(self.rect)
        self.rect = None

    def undo_box(self):
        """
        Clear the previously drawn box
        """
        if len(self.NEW_BOXES) > 0:
            self.canvas.delete(self.NEW_BOXES[-1])
            del self.NEW_BOXES[-1]

    def clear_all_boxes(self):
        """
        Clear every drawn box from the canvas
        """
        for i in range(len(self.NEW_BOXES)):
            self.canvas.delete(self.NEW_BOXES[i])
        self.NEW_BOXES = []


class TkVerifyWords(tk.Tk):
    def __init__(self, words_df, chars_df):
        global ORIGINAL_IMAGE
        global SCALE_FACTOR

        super().__init__()

        self.title("Tesseract Text Correction")
        self.toplevel = None

        self.cropped_img = None
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.BOUNDING_BOXES = []
        self.SUBCONTAINERS = []

        # For each entry in words_df, extract the subimage and scale it down for display,
        # building up a list of word-image pairs
        self.words_df = words_df
        self.chars_df = chars_df
        self.chars_df_backup = self.chars_df.copy()
        self.initial_chars_df_backup = self.chars_df.copy()
        self.chars_df_individual_word_backup = {}
        self.cur_word = 0
        self.WORD_IMAGE_PAIRS = []
        self.MODIFIED_WORDS = set()
        self.FLAGGED_WORDS = []
        needed_width = 0
        needed_height = 0
        for word in self.words_df.itertuples():
            cropped_img = ORIGINAL_IMAGE.crop((word.left, word.top, word.right, word.bottom))
            maxsize = (max(cropped_img.width / SCALE_FACTOR, 1), max(cropped_img.height / SCALE_FACTOR, 1))
            cropped_img.thumbnail(maxsize, Image.ANTIALIAS)
            self.WORD_IMAGE_PAIRS.append([word.value, cropped_img.copy(), word.Index])
            if word.width / SCALE_FACTOR > needed_width:
                needed_width = word.width / SCALE_FACTOR
        needed_width += 400  # Add pixel buffer for text fields

        self.edit_container = None
        self.edit_container_ref = None
        self.edit_canvas = None
        self.edit_vbar = None
        self.canvas = tk.Canvas(self)
        tk.Button(self, text='Finished', command=self.finish).pack(side=tk.RIGHT)
        tk.Button(self, text='Reset', command=self.reset_text).pack(side=tk.RIGHT)
        container = tk.Frame(self.canvas, width=needed_width, height=needed_height)
        self.TEXT_REFS = {}
        for word_image_pair in self.WORD_IMAGE_PAIRS:
            bottom = tk.Frame(container)
            bottom.pack(side=tk.TOP, pady=10)
            tx = tk.Label(self, text=str(word_image_pair[0]))
            img = ImageTk.PhotoImage(word_image_pair[1])
            panel = tk.Label(self, image=img)
            panel.image = img
            btn = tk.Button(self, text='Edit', command=partial(self.edit_text, word_image_pair[2]))
            if self.words_df.at[word_image_pair[2], 'flag']:
                char_slice_df = self.chars_df[self.chars_df['word'] == word_image_pair[2]]
                if char_slice_df.shape[0] > 0:
                    rx = tk.Label(self, text=str("".join(char_slice_df['symbol'].values.tolist())), foreground="red")
                else:
                    rx = tk.Label(self, text=str(word_image_pair[0]), foreground="red")
                self.FLAGGED_WORDS.append(word_image_pair[2])
                self.MODIFIED_WORDS = self.MODIFIED_WORDS.union({word_image_pair[2]})
            else:
                rx = tk.Label(self, text="", foreground="red")
            tx.pack(in_=bottom, side=tk.LEFT, padx=10)
            panel.pack(in_=bottom, side=tk.LEFT, padx=10)
            rx.pack(in_=bottom, side=tk.RIGHT, padx=10)
            btn.pack(in_=bottom, side=tk.RIGHT, padx=10)
            self.TEXT_REFS[word_image_pair[2]] = [tx, btn, rx]
            self.chars_df_individual_word_backup[word_image_pair[2]] = self.chars_df[
                self.chars_df['word'] == word_image_pair[2]].copy()
            bottom.update()
            needed_height += bottom.winfo_height()
        self.canvas.create_window((needed_width, needed_height), window=container, anchor='nw')
        vbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=self.canvas.yview)
        hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.canvas.configure(xscrollcommand=hbar.set)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.canvas.xview)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.wm_geometry("{}x{}".format(int(needed_width), 600))

    def reset_text(self):
        self.chars_df = self.initial_chars_df_backup.copy()
        self.MODIFIED_WORDS = set()
        self.MODIFIED_WORDS = self.MODIFIED_WORDS.union(set(self.FLAGGED_WORDS))
        for word_image_pair in self.WORD_IMAGE_PAIRS:
            self.TEXT_REFS[word_image_pair[2]][0]['text'] = str(word_image_pair[0])
            if self.words_df.at[word_image_pair[2], 'flag']:
                self.TEXT_REFS[word_image_pair[2]][2]['text'] = str(word_image_pair[0])
            else:
                self.TEXT_REFS[word_image_pair[2]][2]['text'] = ""

    def edit_text(self, words_df_index):
        """
        Pop-up window which asks the user to indicate the bounds of each
        character in the targeted word
        """
        self.chars_df_backup = self.chars_df[self.chars_df['word'] == words_df_index].copy()
        self.cur_word = words_df_index
        self.toplevel = tk.Toplevel()
        self.toplevel.wm_protocol("WM_DELETE_WINDOW", self.on_edit_window_close)
        self.edit_canvas = tk.Canvas(self.toplevel)
        self.edit_canvas.pack()

        tk.Button(self.edit_canvas, text='Finished', command=self.update_chars).pack(side=tk.RIGHT)
        tk.Button(self.edit_canvas, text='Reset', command=self.reset_chars).pack(side=tk.RIGHT)

        # Crop the subimage from the full-sized image, add it to the canvas, and make it interactive
        self.cropped_img = ORIGINAL_IMAGE.crop(
            (self.words_df.at[words_df_index, 'left'], self.words_df.at[words_df_index, 'top'],
             self.words_df.at[words_df_index, 'right'], self.words_df.at[words_df_index, 'bottom']))
        cropped_img_ref = ImageTk.PhotoImage(self.cropped_img)
        img_ref = self.edit_canvas.create_image(self.cropped_img.size[0] / 2, self.cropped_img.size[1] / 2,
                                                image=cropped_img_ref)
        self.edit_canvas.image = cropped_img_ref

        # Add the container for storing each character text entry box and its associated button
        self.edit_container = tk.Frame(self.edit_canvas)
        needed_width, needed_height = self.draw_char_edit_frames(words_df_index)

        # Build the display
        self.build_edit_canvas_display(needed_height)

        # Set up the mouse event bindings
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.edit_canvas.tag_bind(img_ref, "<ButtonPress-1>", self.on_button_press)
        self.edit_canvas.tag_bind(img_ref, "<B1-Motion>", self.on_move_press)
        self.edit_canvas.tag_bind(img_ref, "<ButtonRelease-1>", self.on_button_release)

        self.edit_canvas.pack(fill=tk.BOTH, expand=1)
        self.toplevel.wm_geometry("{}x{}".format(int(needed_width), 600))

    def draw_char_edit_frames(self, words_df_index):
        self.cropped_img = ORIGINAL_IMAGE.crop(
            (self.words_df.at[words_df_index, 'left'], self.words_df.at[words_df_index, 'top'],
             self.words_df.at[words_df_index, 'right'], self.words_df.at[words_df_index, 'bottom']))
        self.BOUNDING_BOXES = []
        self.SUBCONTAINERS = []
        needed_width = self.cropped_img.size[0] + 300
        needed_height = self.cropped_img.size[1]
        try:
            chars_df_slice = self.chars_df[self.chars_df['word'] == words_df_index]
            chars_df_slice = chars_df_slice.sort_values(['left'])
        except:
            chars_df_slice = pd.DataFrame()
        for char in chars_df_slice.itertuples():
            # Build the text input for this character
            subcontainer = tk.Frame(self.edit_container)
            self.SUBCONTAINERS.append(subcontainer)
            subcontainer.pack(side=tk.TOP, pady=10)
            tx = CustomText(self.edit_canvas, height=1, width=15)
            tx.insert(tk.END, char.symbol)
            tx.bind("<<TextModified>>", partial(self.on_char_change, char.Index))
            tx.pack(in_=subcontainer, side=tk.LEFT, padx=10)

            # Scale and draw the rectangle on the canvas
            char_df = pd.DataFrame([[char.left, char.top, char.right, char.bottom]],
                                   columns=["xmin", "ymin", "xmax", "ymax"])
            char_df['xmin'] = char_df['xmin'] - self.words_df.at[words_df_index, 'left']
            char_df.loc[char_df.xmin < 0, 'xmin'] = 0
            char_df['xmax'] = char_df['xmax'] - self.words_df.at[words_df_index, 'left']
            char_df.loc[char_df.xmax > self.cropped_img.size[0], 'xmax'] = self.cropped_img.size[0]
            char_df['ymin'] = char_df['ymin'] - self.words_df.at[words_df_index, 'top']
            char_df.loc[char_df.ymin < 0, 'ymin'] = 0
            char_df['ymax'] = char_df['ymax'] - self.words_df.at[words_df_index, 'top']
            char_df.loc[char_df.ymax > self.cropped_img.size[1], 'ymax'] = self.cropped_img.size[1]
            rect = self.edit_canvas.create_rectangle(char_df.iloc[0].xmin, char_df.iloc[0].ymin, char_df.iloc[0].xmax,
                                                     char_df.iloc[0].ymax, outline='red')
            self.BOUNDING_BOXES.append(rect)
            subcontainer.update()
            needed_height += subcontainer.winfo_height()

            # Add the removal button
            btn = tk.Button(self.edit_canvas, text='Remove')
            btn['command'] = partial(self.remove_char, char.Index)
            btn.pack(in_=subcontainer, side=tk.RIGHT, padx=10)

        return needed_width, needed_height

    def build_edit_canvas_display(self, needed_height):
        self.edit_container_ref = self.edit_canvas.create_window((0, needed_height), window=self.edit_container,
                                                                 anchor='nw')
        self.edit_vbar = tk.Scrollbar(self.edit_canvas, orient=tk.VERTICAL)
        self.edit_canvas.configure(yscrollcommand=self.edit_vbar.set)
        self.edit_vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.edit_vbar.config(command=self.edit_canvas.yview)
        self.edit_canvas.configure(scrollregion=self.edit_canvas.bbox("all"))

    def on_char_change(self, char_index, event):
        self.chars_df.at[char_index, 'symbol'] = event.widget.get("1.0", tk.END).rstrip('\r\n')

    def on_button_press(self, event):
        # Save mouse drag start position
        self.start_x = self.edit_canvas.canvasx(event.x)
        self.start_y = self.edit_canvas.canvasy(event.y)
        self.rect = self.edit_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                      outline='red')

    def on_move_press(self, event):
        cur_x = self.edit_canvas.canvasx(event.x)
        cur_y = self.edit_canvas.canvasy(event.y)

        # Expand rectangle as you drag the mouse
        self.edit_canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        # Save this rectangle to the list of current custom rectangles
        self.BOUNDING_BOXES.append(self.rect)

        # Add the new text input field and redraw the modification frames
        xmin, ymin, xmax, ymax = self.edit_canvas.coords(self.rect)
        new_char_df = pd.DataFrame(
            [['', xmin + self.words_df.at[self.cur_word, 'left'], ymax + self.words_df.at[self.cur_word, 'top'],
              xmax + self.words_df.at[self.cur_word, 'left'], ymin + self.words_df.at[self.cur_word, 'top'], 0,
              self.cur_word]],
            columns=["symbol", "left", "bottom", "right", "top", "page", "word"])
        self.chars_df = self.chars_df.append(new_char_df, sort=False)
        self.chars_df.sort_values(['word', 'left'], inplace=True)
        self.chars_df.reset_index(inplace=True, drop=True)
        for rect_ref in self.BOUNDING_BOXES:
            self.edit_canvas.delete(rect_ref)
        for subcontainer in self.SUBCONTAINERS:
            subcontainer.destroy()
        self.draw_char_edit_frames(self.cur_word)
        self.edit_container.update()
        self.edit_canvas.configure(scrollregion=self.edit_canvas.bbox("all"))

        self.rect = None

    def remove_char(self, chars_df_index):
        for rect_ref in self.BOUNDING_BOXES:
            self.edit_canvas.delete(rect_ref)
        for subcontainer in self.SUBCONTAINERS:
            subcontainer.destroy()
        self.chars_df.drop(chars_df_index, inplace=True)
        self.chars_df.reset_index(inplace=True, drop=True)
        self.draw_char_edit_frames(self.cur_word)
        self.edit_container.update()
        self.edit_canvas.configure(scrollregion=self.edit_canvas.bbox("all"))

    def reset_chars(self):
        for rect_ref in self.BOUNDING_BOXES:
            self.edit_canvas.delete(rect_ref)
        for subcontainer in self.SUBCONTAINERS:
            subcontainer.destroy()
        self.chars_df.drop(self.chars_df[self.chars_df.word == self.cur_word].index, inplace=True)
        self.chars_df = self.chars_df.append(self.chars_df_individual_word_backup[self.cur_word], sort=False)
        self.chars_df.sort_values(['word', 'left'], inplace=True)
        self.chars_df.reset_index(inplace=True, drop=True)
        self.draw_char_edit_frames(self.cur_word)
        self.edit_container.update()

    def on_edit_window_close(self):
        self.chars_df.drop(self.chars_df[self.chars_df.word == self.cur_word].index, inplace=True)
        self.chars_df = self.chars_df.append(self.chars_df_backup, sort=False)
        self.chars_df.sort_values(['word', 'left'], inplace=True)
        self.chars_df.reset_index(inplace=True, drop=True)
        self.toplevel.destroy()

    def update_chars(self):
        if not np.array_equal(self.chars_df[self.chars_df['word'] == self.cur_word].values,
                              self.chars_df_individual_word_backup[self.cur_word].values):
            self.TEXT_REFS[self.cur_word][2]['text'] = "".join(
                self.chars_df[self.chars_df.word == self.cur_word]['symbol'].tolist())
            self.MODIFIED_WORDS = self.MODIFIED_WORDS.union({self.cur_word})
        else:
            self.TEXT_REFS[self.cur_word][2]['text'] = ""
            self.MODIFIED_WORDS = self.MODIFIED_WORDS - {self.cur_word}
        self.toplevel.destroy()

    def finish(self):
        global RUNTIME_ID
        global OUTPUT_DIR
        global OUTPUT_DIR_PATH

        # Generate the line-box files
        for i, words_df_index in enumerate(self.MODIFIED_WORDS):
            cropped_img = ORIGINAL_IMAGE.crop(
                (self.words_df.at[words_df_index, 'left'], self.words_df.at[words_df_index, 'top'],
                 self.words_df.at[words_df_index, 'right'], self.words_df.at[words_df_index, 'bottom']))
            chars_df_slice = self.chars_df[self.chars_df['word'] == words_df_index].copy()
            chars_df_slice['left'] = chars_df_slice['left'] - self.words_df.at[words_df_index, 'left']
            chars_df_slice.loc[chars_df_slice.left < 0, 'left'] = 0
            chars_df_slice['right'] = chars_df_slice['right'] - self.words_df.at[words_df_index, 'left']
            chars_df_slice.loc[chars_df_slice.right > cropped_img.size[0], 'right'] = cropped_img.size[0]
            chars_df_slice['top'] = chars_df_slice['top'] - self.words_df.at[words_df_index, 'top']
            chars_df_slice.loc[chars_df_slice.top < 0, 'top'] = 0
            chars_df_slice['bottom'] = chars_df_slice['bottom'] - self.words_df.at[words_df_index, 'top']
            chars_df_slice.loc[chars_df_slice.bottom > cropped_img.size[1], 'bottom'] = cropped_img.size[1]
            output = ""
            for char in chars_df_slice.itertuples():
                # Note that the top and bottom values are swapped. This is because, although y values generally grow
                # downward in image libraries, Tesseract treats them as growing upward for training purposes.
                left = char.left
                bottom = cropped_img.size[1] - char.bottom
                right = char.right
                top = cropped_img.size[1] - char.top
                output += char.symbol + " " + str(int(left)) + " " + str(int(bottom)) + " " + str(
                    int(right)) + " " + str(int(top)) + " 0\n"
            output = output.rstrip("\n")
            img_path = os.path.join(os.path.dirname(OUTPUT_DIR_PATH), OUTPUT_DIR,
                                    'image_' + RUNTIME_ID + '_' + str(i) + '.tif')
            box_path = os.path.join(os.path.dirname(OUTPUT_DIR_PATH), OUTPUT_DIR,
                                    'image_' + RUNTIME_ID + '_' + str(i) + '.box')
            cropped_img.save(img_path, dpi=(STANDARD_DPI, STANDARD_DPI))
            with open(box_path, 'w') as box_file:
                print(output, file=box_file)
        self.destroy()


class CustomText(tk.Text):
    def __init__(self, *args, **kwargs):
        """
        A text widget that report on internal widget commands
        """
        tk.Text.__init__(self, *args, **kwargs)

        # Create a proxy for the underlying widget
        self._orig = self._w + "_orig"
        self.tk.call("rename", self._w, self._orig)
        self.tk.createcommand(self._w, self._proxy)

    def _proxy(self, command, *args):
        cmd = (self._orig, command) + args
        result = self.tk.call(cmd)

        if command in ("insert", "delete", "replace"):
            self.event_generate("<<TextModified>>")

        return result


class UserInputs(object):
    total_files_added = []

    def __init__(self):
        self.files_to_parse = []
        self.out_dict = {}

    def reset(self):
        self.files_to_parse = []
        self.out_dict = {}

    def enquire_file_to_parse(self, dialog=True):
        global kill
        fulfilled_mission = False
        tries = 0
        while not fulfilled_mission and not kill:
            tries += 1
            if tries == 2:
                kill = True
                break
            try:
                file_selected = False
                fp = None
                while not file_selected:
                    if dialog:
                        root = tk.Tk()
                        fp = askopenfilename(initialdir=os.getcwd(), title="Select PDF file",
                                             filetypes=(("PDF files", "*.pdf"),))
                        file_selected = True if len(fp) > 1 else False
                        root.update()
                        root.destroy()
                    else:
                        fp = str(input("Give PDF file to load: "))
                        file_selected = True if len(fp) > 1 else False
                    print(fp)
                error_message = "This is not a PDF file"
                pdf_files = []

                # Check if is file or dir
                page = None
                if os.path.isfile(fp):
                    while page is None or not page.isdigit():
                        page = input("Please select a page to process: ")
                    print("Processing page {}".format(page))
                    if fp.lower().endswith(".pdf"):
                        pdf_files.append(fp)
                else:
                    print("That is not a valid PDF file path")

                if len(pdf_files) < 1:
                    print(error_message)
                    raise error_message
                else:
                    self.files_to_parse.extend(pdf_files)
                    UserInputs.total_files_added.extend(pdf_files)
                    self.out_dict.update({"file": self.files_to_parse, "page": page})
                    fulfilled_mission = True
            except KeyboardInterrupt:
                kill = True
                break
            except Exception as e:
                print(e)
                pass


def generate_unicharset_file():
    global OUTPUT_DIR
    global OUTPUT_DIR_PATH

    # Generate the all-boxes file
    all_boxes_cmd = os.path.dirname(OUTPUT_DIR_PATH) + "/" + OUTPUT_DIR + "/*.box > " + os.path.dirname(
        OUTPUT_DIR_PATH) + "/all-boxes"
    if os.name != 'nt':
        all_boxes_cmd = "cat " + all_boxes_cmd
        process = subprocess.Popen([all_boxes_cmd], shell=True)
        process.wait()
    else:
        all_boxes_cmd = "type " + all_boxes_cmd
        all_boxes_cmd = all_boxes_cmd.replace("/", "\\")
        os.system(all_boxes_cmd)

    # Generate the unicharset file
    unicharset_cmd = "ruby extract_unicharset.rb " + os.path.dirname(OUTPUT_DIR_PATH) + "/all-boxes > "
    unicharset_cmd += os.path.dirname(OUTPUT_DIR_PATH) + "/unicharset"
    if os.name != 'nt':
        process = subprocess.Popen([unicharset_cmd], shell=True)
        process.wait()
    else:
        unicharset_cmd = unicharset_cmd.replace("/", "\\")
        os.system(unicharset_cmd)


def generate_lstmf_files(lang='eng'):
    global OUTPUT_DIR
    global OUTPUT_DIR_PATH
    global LSTMF_FOLDER
    global TESSDATA_FOLDER

    # Generate an LSTMF file for each line-box image in the directory
    tif_files = glob.glob(os.path.join(os.path.dirname(OUTPUT_DIR_PATH), OUTPUT_DIR, '*.tif'))
    lstm_dir = os.path.join(os.path.dirname(OUTPUT_DIR_PATH), LSTMF_FOLDER)
    pathlib.Path(lstm_dir).mkdir(parents=True, exist_ok=True)
    traineddata_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), lang, TESSDATA_FOLDER,
                                        lang + '.traineddata')
    traineddata_dir = os.path.dirname(traineddata_location)
    for tif_file in tif_files:
        file_path = os.path.abspath(tif_file)
        base_name = tif_file[0:tif_file.rfind('.')]
        if os.name != 'nt':
            cmd = "TESSDATA_PREFIX={} tesseract {} {} --psm 6 {}/lstm.train".format(traineddata_dir, file_path,
                                                                                    base_name, lstm_dir)
            process = subprocess.Popen([cmd], shell=True)
            process.wait()
        else:
            traineddata_dir = traineddata_dir.replace("/", "\\")
            file_path = file_path.replace("/", "\\")
            base_name = base_name.replace("/", "\\")
            lstm_dir = lstm_dir.replace("/", "\\")
            cmd = 'cmd /V /C "set TESSDATA_PREFIX={}&& tesseract {} {} --psm 6 {}\\lstm.train"'.format(traineddata_dir,
                                                                                                       file_path,
                                                                                                       base_name,
                                                                                                       lstm_dir)
            os.system(cmd)

    # Generate a singular all-lstmf file, randomly sorted for training purposes
    if os.name != 'nt':
        cmd = "find `pwd` -name '*.lstmf' | sort -R > {}/all-lstmf".format(lstm_dir)
        process = subprocess.Popen([cmd], shell=True)
        process.wait()
    else:
        cmd = "dir /S /b *.lstmf | shuffle.bat > {}\\all-lstmf".format(lstm_dir)
        os.system(cmd)

    # Return the number of line-box images used to generate the LSTMF files
    return len(tif_files)


def generate_training_and_evaluation_files(num_lstmf_files):
    global OUTPUT_DIR_PATH
    global LSTMF_FOLDER

    # Generate the list of training and evaluation files
    holdout = max(int(num_lstmf_files * 0.33), 1)
    lstm_dir = os.path.join(os.path.dirname(OUTPUT_DIR_PATH), LSTMF_FOLDER)
    if os.name != 'nt':
        cmd = "head -n {} {}/all-lstmf > {}/list.eval".format(holdout, lstm_dir, lstm_dir)
        process = subprocess.Popen([cmd], shell=True)
        process.wait()
        cmd = "tail -n +{} {}/all-lstmf > {}/list.train".format(holdout + 1, lstm_dir, lstm_dir)
        process = subprocess.Popen([cmd], shell=True)
        process.wait()
    else:
        lstm_dir = lstm_dir.replace("/", "\\")
        cmd = 'powershell -command "& {get-content ' + lstm_dir + '\\all-lstmf|select-object -first ' + str(
            holdout) + '}" > '
        cmd += lstm_dir + '\\list.eval'
        os.system(cmd)
        cmd = "more +{} {}\\all-lstmf > {}\\list.train".format(holdout, lstm_dir, lstm_dir)
        os.system(cmd)


def train_tesseract_model(traineddata_location):
    global OUTPUT_DIR_PATH
    global LSTMF_FOLDER
    global TESSDATA_FOLDER

    lstm_dir = os.path.join(os.path.dirname(OUTPUT_DIR_PATH), LSTMF_FOLDER)
    lstm_checkpoint = os.path.join(lstm_dir, 'checkpoint.lstm')

    cmd = "combine_tessdata -e {} {}".format(traineddata_location, lstm_checkpoint)
    if os.name != 'nt':
        process = subprocess.Popen([cmd], shell=True)
        process.wait()
    else:
        cmd = cmd.replace("/", "\\")
        os.system(cmd)

    cmd = "lstmtraining "
    cmd += "--model_output {} ".format(os.path.join(os.path.dirname(traineddata_location), TESSDATA_FOLDER))
    cmd += "--continue_from {} ".format(lstm_checkpoint)
    cmd += "--traineddata {} ".format(traineddata_location)
    cmd += "--train_listfile {} ".format(os.path.join(lstm_dir, 'list.train'))
    cmd += "--eval_listfile {}".format(os.path.join(lstm_dir, 'list.eval'))
    if os.name != 'nt':
        process = subprocess.Popen([cmd], shell=True)
        process.wait()
    else:
        cmd = cmd.replace("/", "\\")
        os.system(cmd)

    # The model has been trained, so combine the new model with the old model and clean up the folders
    cmd = "lstmtraining --stop_training "
    cmd += "--continue_from {} ".format(
        os.path.join(os.path.dirname(traineddata_location), TESSDATA_FOLDER + "_checkpoint"))
    cmd += "--traineddata {} ".format(traineddata_location)
    cmd += "--model_output {}".format(traineddata_location)
    if os.name != 'nt':
        process = subprocess.Popen([cmd], shell=True)
        process.wait()
        cmd = "rm {}*".format(os.path.join(os.path.dirname(traineddata_location), TESSDATA_FOLDER))
        process = subprocess.Popen([cmd], shell=True)
        process.wait()
    else:
        cmd = cmd.replace("/", "\\")
        os.system(cmd)
        cmd = "del /S /F {}*".format(
            os.path.join(os.path.dirname(traineddata_location), TESSDATA_FOLDER)).replace("/", "\\")
        os.system(cmd)


def run_postprocessing_functions(lang='eng'):
    traineddata_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), lang, lang + '.traineddata')

    print("Generating the unicharset file...")
    generate_unicharset_file()

    print("Generating the LSTMF files...")
    num_lstmf_files = generate_lstmf_files(lang=lang)

    print("Generating the training and evaluation files...")
    generate_training_and_evaluation_files(num_lstmf_files=num_lstmf_files)

    print("Retraining Tesseract. This could take a while...")
    train_tesseract_model(traineddata_location=traineddata_location)

    print("Finished retraining. Model saved to: {}".format(traineddata_location))


def print_help():
    print("\nUsage: python pipeline.py [-l [<language>|eng]] [-b] [-p] [-n] [-f] [-r] [-h]\n")
    print("    -l, --language <eng>    The three-letter code of the language of the model to retrain")
    print("    -b, --linebox           Run the GUI to generate line-box images")
    print("    -p, --postprocessing    Only run the post-processing script suite")
    print("    -n, --unicharset        Only run unicharset file generation")
    print("    -f, --lstmf             Only generate LSTMF training and evaluation files from the line-box images")
    print("    -r, --retrain           Only retrain the Tesseract model from the line-box and LSTMF files")
    print("    -h, --help              Display detailed help output (this dialog)\n")
    print("To run the pipeline from start to finish, start by building all the line-box images:")
    print("    python pipeline.py")
    print("\nOnce all line-box images from all files have been made (you need at least 3), run the following " +
          "commands in order:")
    print("    python pipeline.py -d")
    print("    python pipeline.py -n")
    print("    python pipeline.py -f")
    print("    python pipeline.py -r")
    print("    python pipeline.py -u")
    print("\nOr, if you want to execute all five commands at once, then just run:")
    print("    python pipeline.py -p")


def main(args, lang='eng'):
    global kill
    global TESSDATA_FOLDER
    global RUNTIME_ID
    global OUTPUT_DIR
    global OUTPUT_DIR_PATH

    if os.name == "nt":
        freeze_support()
        set_start_method('spawn', force=True)  # Pretend to be Windows
    kill = False
    first_run = True
    files_processed_counter = 0

    OUTPUT_DIR_PATH = os.path.abspath(lang)
    if not os.path.exists(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)
    OUTPUT_DIR_PATH = os.path.join(os.path.dirname(OUTPUT_DIR_PATH), lang, OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)

    if args['help']:
        print_help()
    elif args['postprocessing']:
        run_postprocessing_functions(lang=lang)
    elif args['unicharset']:
        print("Generating the unicharset file...")
        generate_unicharset_file()
    elif args['lstmf']:
        print("Generating the LSTMF files...")
        num_lstmf_files = generate_lstmf_files(lang=lang)
        print("Generated", str(num_lstmf_files), "LSTMF files")
        if num_lstmf_files is None or num_lstmf_files <= 0:
            print("No training or evaluation files to generate, exiting")
        else:
            print("Generating the training and evaluation files...")
            generate_training_and_evaluation_files(num_lstmf_files=num_lstmf_files)
    elif args['retrain']:
        traineddata_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), lang, TESSDATA_FOLDER,
                                            lang + '.traineddata')
        print("Retraining Tesseract. This could take a while...")
        train_tesseract_model(traineddata_location=traineddata_location)
    else:
        while not kill:
            user_input = UserInputs()
            user_input.reset()  # Resets user inputs on each loop
            if not first_run:
                print("All files processed so far: {}".format(
                    [os.path.basename(base_name) for base_name in UserInputs.total_files_added]))
                print("Total files processed so far: {}".format(len(UserInputs.total_files_added)))
                print("Total files processed just now: {}".format(files_processed_counter))
            print("\n")

            # Ask for user inputs here
            user_input.enquire_file_to_parse(dialog=True)
            for file in user_input.out_dict['file']:
                RUNTIME_ID = str(int(time())) + str(random.randint(-sys.maxsize - 1, sys.maxsize))
                tkgui = TkDrawBorders(pdf_path=file, pg=user_input.out_dict['page'])
                tkgui.mainloop()

            first_run = False
            files_processed_counter = len(user_input.files_to_parse)
            continue_input = input("Press y to continue: ")

            if not (continue_input == 'y' or continue_input == 'Y'):
                # User did not enter 'y', so stop
                kill = True


if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-l", "--language", type=str, default="eng",
                    help="The Tesseract label of the language of the model to retrain")
    ap.add_argument("-b", "--linebox", action="store_true", help="Run the GUI to generate line-box images")
    ap.add_argument("-p", "--postprocessing", action="store_true", help="Only run the post-processing script suite")
    ap.add_argument("-n", "--unicharset", action="store_true", help="Only run unicharset file generation")
    ap.add_argument("-f", "--lstmf", action="store_true",
                    help="Only generate LSTMF training and evaluation files from the line-box images")
    ap.add_argument("-r", "--retrain", action="store_true",
                    help="Only retrain the Tesseract model from the line-box and LSTMF files")
    ap.add_argument("-h", "--help", action="store_true", help="Display detailed help output")
    args = vars(ap.parse_args())
    kill = False
    main(args, lang=args["language"])
