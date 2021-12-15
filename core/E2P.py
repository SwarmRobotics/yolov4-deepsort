import os
import sys
import cv2
import numpy as np
from math import pi
from matplotlib import pyplot as plt

class E2P: 
    def __init__(self, frame_w, frame_h, wFOV, hFOV, THETA, PHI, height, width, RADIUS=1):
        # Setup attributes
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.wFOV = wFOV
        self.hFOV = hFOV
        self.THETA = THETA
        self.PHI = PHI
        self.height = height
        self.width = width
        self.RADIUS = RADIUS

        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0

        #Longtitute = around the circule = PHi (irl) = THETA
        #lATITUDE = up and down the circule = LAMBDA (irl) = PHI

        self.cp = [np.deg2rad(self.PHI), np.deg2rad(self.THETA)]
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_cx = (self.frame_w - 1) / 2.0
        equ_cy = (self.frame_h - 1) / 2.0

        #wFOV = FOV
        #hFOV = float(height) / width * wFOV
        

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        self.lon = lon / 180 * equ_cx + equ_cx
        self.lat = lat / 90 * equ_cy + equ_cy
        #for x in range(width):
        #    for y in range(height):
        #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
        #return self._img 

    def Remap (self, frame):
        return cv2.remap(frame, self.lon.astype(np.float32), self.lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    def Rect2Sphere(self, coord):
        x = 2*coord[0]/self.width- 1
        y = 2*coord[1]/self.height - 1

        x = x * np.tan((np.radians(self.wFOV/2)))
        y = y * np.tan((np.radians(self.hFOV/2)))
        
        lat = np.arctan(y) + np.deg2rad(self.PHI)
        lon = np.arctan(x) + np.deg2rad(self.THETA)

        return [lon, lat]

    def Rect2SphereArray(self, coord):
        output = []
        for row in coord:
            ymin = (2 * row[0]) - 1
            xmin = (2 * row[1]) - 1
            ymax = (2 * row[2]) - 1
            xmax = (2 * row[3]) - 1

            ymin = ymin * np.tan((np.radians(self.hFOV / 2)))
            xmin = xmin * np.tan((np.radians(self.wFOV / 2)))
            ymax = ymax * np.tan((np.radians(self.hFOV / 2)))
            xmax = xmax * np.tan((np.radians(self.wFOV / 2)))

            lonmin = np.arctan(ymin) + np.deg2rad(self.PHI)
            latmin = np.arctan(xmin) + np.deg2rad(self.THETA)
            lonmax = np.arctan(ymax) + np.deg2rad(self.PHI)
            latmax = np.arctan(xmax) + np.deg2rad(self.THETA)

            output.append([lonmin, latmin, lonmax, latmax])
        return output

    def Sphere2Point(self, coord):
        
        #Convert to relative angles
        u = coord[0]/self.PI2
        v = coord[1]/self.PI
        
        #Convert to normalised coordinates
        x = (u + 0.5)*self.frame_w
        y = (v + 0.5)*self.frame_h        
        return [x, y]

    def Sphere2PointArray(self, coord):
        output = []
        for row in coord:
            # Convert to relative angles
            ymin = row[0] / self.PI
            xmin = row[1] / self.PI2
            ymax = row[2] / self.PI
            xmax = row[3] / self.PI2

            # Convert to normalised coordinates
            ymin = (ymin + 0.5)
            xmin = (xmin + 0.5)
            ymax = (ymax + 0.5)
            xmax = (xmax + 0.5)

            # Append to output
            output.append([ymin,xmin,ymax,xmax])
        return output

    def format_boxes(self, bboxes):
        for box in bboxes:
            ymin = int(box[0] * self.frame_h)
            xmin = int(box[1] * self.frame_w)
            ymax = int(box[2] * self.frame_h)
            xmax = int(box[3] * self.frame_w)
            width = xmax - xmin
            height = ymax - ymin
            box[0], box[1], box[2], box[3] = ymin, xmin, ymax, xmax
        return bboxes
    
    def Point2Sphere(self, coord):
        #Convert to normalised coordinates
        u = (coord[0]/self.width - 0.5)
        v = (coord[1]/self.height - 0.5)

        #Convert to relative angles
        theta = u*2*pi
        phi = v*pi

        theta = round(theta, 6)
        phi = round(phi, 6)
        #Limits of panramoa are -pi<x<pi and -pi/2<y<pi/2
        
        return [theta, phi]

    def Coord2WorldStatic(coord, w, h):
        #Convert to normalised coordinates
        u = (coord[0]/w - 0.5)
        v = (coord[1]/h - 0.5)

        #Convert to relative angles
        theta = u*2*pi
        phi = v*pi

        theta = round(theta, 6)
        phi = round(phi, 6)
        #Limits of panramoa are -pi<x<pi and -pi/2<y<pi/2
        
        return [theta, phi]

    def clickEvent1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            projectionAngle = self.calcSphericaltoGnomonic2(x,y)
            strXY = str(np.rad2deg(projectionAngle))
            print('Button Clicked at: ' + strXY)
            print('Relative Point:', self.relative_projection_point(projectionAngle[0], projectionAngle[1]))

def clickEvent2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        strXY = str(np.rad2deg(E2P.relative_projection_angle(x, y, 800, 400)))
        print('Button Clicked at: ' + strXY)
        font = cv2.FONT_HERSHEY_SIMPLEX

def main():
    #equ = Equirectangular('example.jpg')    # Load equirectangular image
    
    #
    # FOV unit is degree 
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension 
    #
    # Specify parameters(FOV, theta, phi, height, width)
    height = 400
    width = 800

    img = cv2.imread('Screenshot.png', cv2.IMREAD_COLOR)
    frame_width, frame_height = img.shape[:2]
    print("Width: {}, Height: {}".format(frame_width, frame_height))
    RADIUS = 128

    Perspective1 = E2P(frame_height, frame_width, 114, 70, 0, 0, height, width, RADIUS)
    Perspective2 = E2P(frame_height, frame_width, 114, 70, 90, 0, height, width, RADIUS)
    Perspective3 = E2P(frame_height, frame_width, 114, 70, 180, 0, height, width, RADIUS)
    Perspective4 = E2P(frame_height, frame_width, 114, 70, -90, 0, height, width, RADIUS)
    
    
    cv2.imshow('perspective 1', Perspective1.Remap(img))
    cv2.imshow('perspective 2', Perspective2.Remap(img))
    cv2.imshow('perspective 3', Perspective3.Remap(img))
    cv2.imshow('perspective 4', Perspective4.Remap(img))
    cv2.imshow('Original', cv2.resize(img, tuple([800,400])))
    cv2.setMouseCallback('perspective 1', Perspective1.clickEvent1)
    cv2.setMouseCallback('perspective 2', Perspective2.clickEvent1)
    cv2.setMouseCallback('perspective 3', Perspective3.clickEvent1)
    cv2.setMouseCallback('perspective 4', Perspective4.clickEvent1)
    cv2.setMouseCallback('Original', clickEvent2);
    cv2.waitKey(0) == ord('q')
    cv2.destroyAllWindows()

def betterProjection():
    height = 400
    width = 800

    img = cv2.imread('/Volumes/Data/Renders/snapUGV.png', 0)
    dimensions = img.shape
    RADIUS = 128

    Perspective1 = E2P(dimensions[1], dimensions[0], 114, 70, 0, 0, height, width, RADIUS)
    Perspective2 = E2P(dimensions[1], dimensions[0], 114, 70, 90, 0, height, width, RADIUS)
    Perspective3 = E2P(dimensions[1], dimensions[0], 114, 70, 180, 0, height, width, RADIUS)
    Perspective4 = E2P(dimensions[1], dimensions[0], 114, 70, -90, 0, height, width, RADIUS)
    
    titles = ['Original Image', '0', '90', '180', '270']
    images = [img, Perspective1, Perspective2, Perspective3, Perspective4]

    for i in range(4):
        plt.subplot(2,3, i+1)
        plt.imshow(images[2], 'gray')
        plt.title(titles[i])

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

