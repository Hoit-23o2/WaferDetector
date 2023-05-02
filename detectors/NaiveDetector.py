import cv2
import os
from WaferDetector import *

class NaiveDetector(WaferDetector):
    def __init__(self, **kwargs):
        super().__init__()
        self.pre_dir = kwargs.get("pre_dir", "pre-output")
        self.mid_dir = kwargs.get("mid_dir", "mid-output")
        self.fnl_dir = kwargs.get("fnl_dir", "final-output")
        # create if not exist
        if not os.path.exists(self.pre_dir):
            os.makedirs(self.pre_dir)
        if not os.path.exists(self.mid_dir):
            os.makedirs(self.mid_dir)
        if not os.path.exists(self.fnl_dir):
            os.makedirs(self.fnl_dir)
        self.depth = 0

    @staticmethod
    def line_link_parallel(img, line1, line2, num_pts=20, outer=10, **kwargs):
        x11, y11, x12, y12 = line1[0]
        x21, y21, x22, y22 = line2[0]
        if x12 == x11:
            step1 = (y12 - y11 if y12 > y11 else y11 - y12) / num_pts
            step2 = (y22 - y21 if y22 > y21 else y21 - y22) / num_pts
            for i in range(num_pts):
                y1 = y11 + step1 * i
                x1 = x11 - outer
                y2 = y21 + step2 * i
                x2 = x21 + outer
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), kwargs["color"], kwargs["thickness"], cv2.LINE_AA)
        else:
            # Fill color of the are between two edges with white
            step1 = (x12 - x11 if x12 > x11 else x11 - x12) / num_pts
            step2 = (x22 - x21 if x22 > x21 else x21 - x22) / num_pts
            print(step1, step2)
            for i in range(num_pts):
                x1 = x11 + step1 * i
                y1 = y11 + (y12 - y11) / (x12 - x11) * (x1 - x11)
                x2 = x21 + step1 * i
                y2 = y21 + (y12 - y11) / (x12 - x11) * (x2 - x21)
                cv2.line(img, (int(x1 - outer), int(y1)), (int(x2 + outer), int(y2)), kwargs["color"], kwargs["thickness"], cv2.LINE_AA)
    
    def eliminate_wafer(self, img, wafer_res):
        line1 = wafer_res["line1"]
        line2 = wafer_res["line2"]
        NaiveDetector.line_link_parallel(img, line1, line2, 100, 10, color=(0, 0, 0), thickness=5)
        NaiveDetector.line_link_parallel(img, line1, line2, 30, 10, color=(255, 255, 255), thickness=5)

    def get_candidate_wafer_lines(self, lines: list[list[list]], error: int) -> tuple[list[list[list]], list[list[list]], int]:
        # sort lines by their length
        lines = sorted(lines, key=lambda x: WaferDetector.line_length(x), reverse=True)
        
        # select the top lines
        top_k = 0
        line_lengths = []
        for line in lines:
            line_lengths.append(WaferDetector.line_length(line))
        
        max_length = WaferDetector.line_length(lines[0])
        for line in lines:
            length = WaferDetector.line_length(line)
            if length > max_length - error:
                top_k += 1
            else:
                break
        selected_wafers = lines[:top_k]
        return (line_lengths, selected_wafers, top_k)

    def highlight_wafers(self, img: cv2.Mat, selected_wafers, **kwargs):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        highlight_color = kwargs["color"] if "color" in kwargs else 255
        highlight_thickness = kwargs["thickness"] if "thickness" in kwargs else 4

        for line in selected_wafers:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), highlight_color, highlight_thickness, cv2.LINE_AA)
        
        # print(img)
        img[img != 255] = 0
        img = WaferDetector.edge_erosion(img, 5)
        img = WaferDetector.edge_dilation(img, 5)
        
        return img
    
    def extract_borders_from_wafer(self, selected_wafers, **kwargs):
        # group lines
        wafer_inner_group_gap = kwargs["wafer_inner_group_gap"] if "wafer_inner_group_gap" in kwargs else 50
        wafer_outer_group_gap = kwargs["wafer_outer_group_gap"] if "wafer_outer_group_gap" in kwargs else 1000        
        
        selected_one_side_line = None
        selected_other_side_line = None

        def pack_line(pts1, pts2):
            return [[pts1[0], pts1[1], pts2[0], pts2[1]]]
        
        one_sides = []
        other_sides = []
        one_sides.append(selected_wafers[0])
        one_side_center = WaferDetector.line_center(selected_wafers[0])
        for i in range(1, len(selected_wafers)):
            line = selected_wafers[i]
            center = WaferDetector.line_center(line)
            if WaferDetector.line_length(pack_line(center, one_side_center)) < wafer_inner_group_gap:
                one_sides.append(line)
            else:
                other_sides.append(line)
        
        # choose two closest lines from each side
        for one_side_line in one_sides:
            for other_side_line in other_sides:
                one_side_center = WaferDetector.line_center(one_side_line)
                other_side_center = WaferDetector.line_center(other_side_line)
                gap = WaferDetector.line_length(pack_line(one_side_center, other_side_center))
                if gap < wafer_outer_group_gap:
                    wafer_outer_group_gap = gap
                    selected_one_side_line = one_side_line
                    selected_other_side_line = other_side_line

        return (selected_one_side_line, selected_other_side_line)

    def pre_process(self, img: cv2.Mat, io = False):
        super().pre_process(img)
        
        edges = WaferDetector.edge_detection(img, "sobel", threshold = 0.3)
        
        eroded_edges = edges
        for i in range(3):
            dilated_edges = WaferDetector.edge_dilation(eroded_edges, 10)
            eroded_edges = WaferDetector.edge_erosion(dilated_edges, 10)

        lines = WaferDetector.line_detection(eroded_edges, "hough", minLineLength=100, maxLineGap=10)
        
        line_lengths, selected_wafers, top_k = self.get_candidate_wafer_lines(lines, 100)
        
        res = self.highlight_wafers(img, selected_wafers)
        
        if io:
            img_name = os.path.basename(self.img_path)
            path = os.path.join(self.pre_dir, "pre-" + img_name)
            self.store_img(path, res)

        return (edges, eroded_edges, dilated_edges, lines, line_lengths, selected_wafers, res)

    def process(self, img: cv2.Mat, io = False, **kwargs):
        super().process(img)

        threshold = kwargs["threshold"] if "threshold" in kwargs else 0.1
        error = kwargs["error"] if "error" in kwargs else 50
        further_process = kwargs["further_process"] if "further_process" in kwargs else False
        
        print("threshold: ", threshold)
        print("error: ", error)
        print("further_process: ", further_process)

        edges = WaferDetector.edge_detection(img, "sobel", threshold = threshold)     
        # Linking Line Segments
        if further_process:
            edges = WaferDetector.edge_dilation(edges, 5)
            edges = WaferDetector.edge_erosion(edges, 6)
            edges = WaferDetector.edge_dilation(edges, 5)
        else:
            edges = WaferDetector.edge_dilation(edges, 5)

        lines = WaferDetector.line_detection(edges, "hough", minLineLength=100, maxLineGap=10)

        # we need smaller error for extracting wafers
        line_lengths, selected_wafers, top_k = self.get_candidate_wafer_lines(lines, error)

        border1, border2 = self.extract_borders_from_wafer(selected_wafers)
        if border1 is None or border2 is None:
            return None

        wafer_res = self.extract_wafer_res(border1, border2)
        if wafer_res is None:
            return None
        
        if io:
            img_name = os.path.basename(self.img_path)
            path = os.path.join(self.mid_dir, str(self.depth) + "-" + img_name + "-wafer-res")
            self.store_wafer_res(path, wafer_res)

        return (edges, lines, line_lengths, selected_wafers, wafer_res)
    
    def post_process(self, img: cv2.Mat, **kwargs):
        super().post_process(img)
        
        further_process = False
        wafer_res = kwargs["wafer_res"]
        if wafer_res is None:
            return None

        self.eliminate_wafer(img, wafer_res)

        img_name = os.path.basename(self.img_path)
        path = os.path.join(self.mid_dir, img_name)
        self.store_img(path, img)

        res = self.process(img, False)
        if res is None:
            return (img, further_process)
        
        edges, lines, line_lengths, selected_wafers, new_wafer_res = res

        # check wafer_res again
        if new_wafer_res["direction"] != wafer_res["direction"]:
            further_process = True
            self.depth += 1
            
        return (img, further_process)
