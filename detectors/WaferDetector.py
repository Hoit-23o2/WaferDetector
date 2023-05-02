#!/usr/bin/env python3
import cv2
import numpy as np
import os


class WaferDetector(object):
    def __init__(self) -> None:
        self.img = None
        self.img_path = None

    """ IO related methods """
    def store_img(self, path, img=None):
        if img is None:
            img = self.img
        print("storing image to: ", path)
        cv2.imwrite(path, img)

    def load_img(self, img_path) -> None:
        img = cv2.imread(img_path)
        self.img = img
        self.img_path = os.path.abspath(img_path)
    
    def store_wafer_res(self, path, res: dict):
        with open(path, 'w') as f:
            f.write(str(res))
    
    def load_wafer_res(self, path):
        with open(path, 'r') as f:
            res = eval(f.read())
        return res
    
    """ general edge and line processing methods """
    @staticmethod
    def edge_detection(img: cv2.Mat, operator = "sobel", **kwargs) -> cv2.Mat:
        # convert to grayscale
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            if img.ndim == 2:
                gray = img
            else:
                raise Exception("image is not grayscale")
        
        edges = None
        if operator == 'sobel':
            threshold = kwargs.get("threshold", 0.3)
            # blur the image
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            # apply sobel filter
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
            # combine sobelx and sobely
            sobel = np.sqrt(sobelx**2 + sobely**2)
            # normalize sobel
            sobel = sobel / np.max(sobel)
            # threshold sobel
            sobel[sobel < threshold] = 0
            sobel[sobel >= threshold] = 1
            edges = sobel
        elif operator == 'canny':
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blur, 255 * threshold * 0.4, 255 * threshold, apertureSize = 3) 
        else:
            edges = gray

        return edges
    
    @staticmethod
    def edge_erosion(edges: cv2.Mat, size):
        # enhance using erosion
        kernel = np.ones((size, size),np.uint8)
        if size == 0:
            return edges
        erosion = cv2.erode(edges, kernel, iterations = 1)
        return erosion

    @staticmethod
    def edge_dilation(edges: cv2.Mat, size):
        # enhance using dilation
        kernel = np.ones((size, size),np.uint8)
        if size == 0:
            return edges
        dilation = cv2.dilate(edges, kernel, iterations = 1)
        return dilation
    
    @staticmethod
    def line_detection(edges, model = "hough", **kwargs):
        # convert edges to grayscale
        gray = (edges * 255).astype(np.uint8)
        lines = None
        if model == "hough":
            # hough transform using HoughLineP
            lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100, minLineLength = kwargs["minLineLength"], maxLineGap = kwargs["maxLineGap"])
        elif model == "LSD":
            # hough transform using LSD
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(gray)
        return lines
    
    @staticmethod
    def line_length(line: list[list]):
        x1, y1, x2, y2 = line[0]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    @staticmethod
    def line_angle(line: list[list]):
        x1, y1, x2, y2 = line[0]
        angle = np.rad2deg(np.arctan2(y1 - y2,  x2 - x1))
        return angle
    
    @staticmethod
    def line_center(line: list[list]) -> tuple:
        x1, y1, x2, y2 = line[0]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    """ wafer related methods """
    @staticmethod
    def extract_wafer_res(line1, line2):
        angle1 = WaferDetector.line_angle(line1)
        angle2 = WaferDetector.line_angle(line2)

        if abs(abs(angle1) - abs(angle2)) > 10:
            return None
        
        print(abs(angle1 - angle2))
        print(angle1, angle2)

        direction = "horizontal"
        angle = (angle1 + angle2) / 2
        if abs(angle) < 45:
            direction = "horizontal"
            # horizontal
            x1 = min(line1[0][0], line2[0][0])
            x2 = max(line1[0][2], line2[0][2])
            line1[0][0] = x1
            line1[0][2] = x2
            line2[0][0] = x1
            line2[0][2] = x2 

            x1, y1, x2, y2 = line1[0]
            k = (y2 - y1) / (x2 - x1)
            x1, y1, x2, y2 = line2[0]
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            y1 = int(k * (x1 - center[0]) + center[1])
            y2 = int(k * (x2 - center[0]) + center[1])
            line2[0][1] = y1
            line2[0][3] = y2

            # draw lines
            if line2[0][1] > line1[0][1]:
                upper_edge = line1
                lower_edge = line2
            else:
                upper_edge = line2
                lower_edge = line1
            lower_x1, lower_y1, lower_x2, lower_y2 = lower_edge[0]
            upper_x1, upper_y1, upper_x2, upper_y2 = upper_edge[0]
            saw_line = [[lower_x1, (lower_y1 + upper_y1) // 2, upper_x2, (lower_y2 + upper_y2) // 2]]
            saw_x1, saw_y1, saw_x2, saw_y2 = saw_line[0]
        else:
            direction = "vertical"
            # vertical
            y1 = min(line1[0][1], line2[0][1])
            y2 = max(line1[0][3], line2[0][3])
            line1[0][1] = y1
            line1[0][3] = y2
            line2[0][1] = y1
            line2[0][3] = y2

            x1, y1, x2, y2 = line1[0]
            k = (x2 - x1) / (y2 - y1)
            x1, y1, x2, y2 = line2[0]
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            x1 = int(k * (y1 - center[1]) + center[0])
            x2 = int(k * (y2 - center[1]) + center[0])
            line2[0][0] = x1
            line2[0][2] = x2

            # draw lines
            if line2[0][0] > line1[0][0]:
                left_edge = line1
                right_edge = line2
            else:
                left_edge = line2
                right_edge = line1
            left_x1, left_y1, left_x2, left_y2 = left_edge[0]
            right_x1, right_y1, right_x2, right_y2 = right_edge[0]
            saw_line = [[(left_x1 + right_x1) // 2, left_y1, (left_x2 + right_x2) // 2, right_y2]]
            saw_x1, saw_y1, saw_x2, saw_y2 = saw_line[0]
        
        angle = np.rad2deg(np.arctan2(saw_y1 - saw_y2,  saw_x2 - saw_x1))
        
        wafer_res = {
            "line1": line1,
            "line2": line2,
            "saw_line": saw_line,
            "angle": angle,
            "direction": direction
        }
        
        return wafer_res

    """ processing related methods """
    def pre_process(self, img: cv2.Mat):
        pass

    def process(self, img: cv2.Mat):
        pass

    def post_process(self, img: cv2.Mat):
        pass 

    """ drawing related methods """
    @staticmethod
    def convert_to_gray(img: cv2.Mat):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            if img.ndim == 2:
                gray = img
            else:
                raise Exception("image is not grayscale")
        return gray
    
    @staticmethod
    def artist_draw_wafer(img: cv2.Mat, wafer_res):
        def cv2_drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20): 
            dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5 
            pts= [] 
            for i in np.arange(0,dist,gap): 
                r=i/dist 
                x=int((pt1[0]*(1-r)+pt2[0]*r)+.5) 
                y=int((pt1[1]*(1-r)+pt2[1]*r)+.5) 
                p = (x,y) 
                pts.append(p) 
        
            if style=='dotted': 
                for p in pts: 
                    cv2.circle(img,p,thickness,color,-1) 
            else: 
                s=pts[0] 
                e=pts[0] 
                i=0 
                for p in pts: 
                    s=e 
                    e=p 
                    if i%2==1: 
                        cv2.line(img,s,e,color,thickness) 
                    i+=1 
        x1, y1, x2, y2 = wafer_res["line1"][0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4, cv2.LINE_AA)
        x1, y1, x2, y2 = wafer_res["line2"][0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4, cv2.LINE_AA)
        saw_x1, saw_y1, saw_x2, saw_y2 = wafer_res["saw_line"][0]
        cv2.line(img, (saw_x1, saw_y1), (saw_x2, saw_y2), (0, 255, 0), 2, cv2.LINE_AA)
        
        # draw a horizontal line in the middle of the image
        center = (img.shape[1] // 2, img.shape[0] // 2)
        cv2_drawline(img, (0, center[1]), (img.shape[1], center[1]), (255, 0, 0), 2, "dashed", 10)
        
        # calc angle in degrees
        angle = wafer_res["angle"]
        print("angle: " + str(round(angle, 2)) + "°")
        if wafer_res["direction"] == "horizontal":
            cv2.putText(img, str(round(angle, 2)) + " deg", ((saw_x1 + saw_x2) // 3, (saw_y1 + saw_y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, str(round(angle, 2)) + " deg", ((saw_x1 + saw_x2) // 2, (saw_y1 + saw_y2) // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        return img