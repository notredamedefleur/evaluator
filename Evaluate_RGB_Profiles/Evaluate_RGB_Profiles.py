import os
import sys
#dll_path = r'C:\Test_Data\Rosa\GmgRosaSdk-internal-1.0.0.4776\RosaSdk\bin'
dll_path = r'C:\Test_Data\Rosa\GmgRosaSdk-internal-1.0.0.5579\RosaSdk\bin'
os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

# Add the path to sys.path so Python can find the bindings
sys.path.append(dll_path)


from datetime import datetime
import pytest
from pathlib import Path
import math

import numpy as np
import re  

import subprocess


import PyRosaEngine
from RosaEngine import GamutMappingParameters

from RosaEngine import RosaEngine, Measurement, Profile, ObserverCondition, DeviceLinkProfileCalculationParams, ProfileMetaInfo, PrintProductParams, InkLayer, RgbDeviceLinkProfileCalculationParams, PrinterProfileCalculationParams
#from PyRosaEngine import ColorantSeparation, InkSaving, SeparationRuleParams, OutputColorant, TestchartCreationParams,Size, Rect, InkInfo, OverprintRule, OVP_FORBIDDEN, TQ_high, Lab, TQ_low, TQ_medium, TQ_very_low, TQ_maximum

from RosaEngine import (
    ObserverCondition, PrinterProfileCalculationParams, 
    DeviceLinkProfileCalculationParams, RgbDeviceLinkProfileCalculationParams,
    GamutMappingParameters, SeparationAlgorithmParameters,
    ProfileMetaInfo,
    OverprintRule, OVP_FORBIDDEN,
    PrintProductParams, InkLayer,
     InkSplittingPair, InkInfo, Lab, TQ_high,
    Size, Rect, 
    TQ_high, Lab, TQ_low, TQ_medium, TQ_very_low, TQ_maximum
)

from base64 import decodebytes
import zlib

#import re
import shutil

from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000



def decodeGMGGrid (filepath):
   

    try:
        file = open(filepath, "r")
        read = file.readlines()
        file.close()

        modified = []
        for line in read:
            modified.extend(line.split())

        for statement in modified:
            if statement == "GMG_Grid":
                GMG_Grid = modified[modified.index(statement)+1]

    except FileNotFoundError:
        print("This filepath dose not exist")

  
    


    #b64 = b"Z2xDYR/xiq02Ar14cyu4Z72Sd8H5VUPrQ0ZeZEqwKpuqA/aG6rmgOWupPVPI4b4emNOHMq6Nfr9UR4ro+9Pj6JD9XMcZMn+ozFkbyZhZWcooC9QYKsNc7fS+qaEr644YLUuwrehyGQ/Zh9vcmpE0yA1VOU5eN5lrWhFM9S8LjDuiqNyfKJxJKcDeByN+geFepRe6AL9RO+9+5RvIF3aqwmjmGFUIUejT7vWpw1c9CJzEAtVYkyuD0o8sq94jUfDn+lH8+6m+87qfE/1VFfVIT7dnXk5KC7YAa/NbZXY3W+W6221/+MvfzxBDQULRHtKIlDZLl1WkUocNsTFyKa6HKbawNFvbmFn59nGuBUfUrUtYByofPbl45qU3PzsMjDBJ+p2sJ2C2GqYZ5SxftDdpiOpBmqA4iamuMG/IO8tRcKXFP7lnzSDIiBPwLWHOitjM4CyOLEJhMXdIhmxR11Sx6Tt1A9lf54LaFZJGveEfkUgtHsEskZm0CKop9d95vwtfc0o0igPyYYysiNPSe8WjzELL1ec45UM88yZxnF8T3SdUlO/d+yVO6Dxc3CWQ0Na4MKmdCySBleaWgC1LOHIzdO4g85jt17h0jhOFU532hd3CrRgeIv6Dsn2C1vb4CXuX+ESWLiyRXSakEs/AeUabTp4FRRmWcyxM9CSum7pTTExwyNb8TtzR9A3t3aOzAGaHcndLAlbgIZWKDKIekWLe6FEOkmh0P+su/jrcaNHTqU68ccW0sreUG6OoLo2QeG3kWlYWsMkjjY2uPU+3i6PcHgxX2EQ/vkjXSQbNx7EhIqydGYbiwGbADl5NMH4a1xXZtMIpTc2cQYfvEqkrgNB+KSiPri1tIlHh9NVsR855ieJAltrs4iKXuS2MdD2FiXQk6I9KUU9zipK6J89dvSjimLdhnpwlmcMV92u1AHdjtZCsGT67t10kaOx3Id0SpZCZGs7NeC4n43Mud9KOCPmLWpvFBQJAEUT3BXOAdTeCVTzzgqx8f4XBsCskM2TOdV/qqVdUQQPu/14Xmlqv1d4khEPmQMHDJ4b5QRd1R4ZD8vI6lwNNWXimF59p+OhGeejSSgxPLorPunrPAL114sW3PJ7BJcTDSPc2tV13fHu8Ah2lqXY9tE2zbuBdOWNFRS65B4/PZtUzKCzKkkstlCtRuL5rEOeUdFzLJCbVSHUizST5AcUXFnqJYk6xwn+xilQhG+8owO5wJcFPs3W2hbxJ6srpvZY6Oe86qv2oGiMiPTIOTWozJvJok6jZiyA9K5I9BtMxryWHd3q1gZItpY22yth/A8Cva1bUh4MdDgsK/FIueyjSLviloyqZQ96lME213JmuyhtV5X+XZFmBJEV59TyvNwy9KNylN7bTu208ximVNOLYLiZHPcMMR8gkKhNp7F2RcqRZb4WwTGvn82UtYkX1UJrywbDDIYJUoStJZHcR5YY2ELaUEVx2I3vVFXV/zXn5XMVKFieJP07swiKx1yR8G7Ione4tJZxP7nXrheFJt8q0vcs6ZO9nqqCoRyN/PW8OEGpupvB2MLESa3G2AGovikVvvlIV5f/zOGHjDRWMmKdhGEyfyyFYR3Z8sSr/IIELP0qkhXlbxDVI1eT59DrVLJUX+llp+OIjwG5TphcRZaKTHyZ/uiDWD4d1B/xGNkxNBCRTzCBaBG6U5llrNIQrE2nsXQl0pFlvhbBMq+bzZS1iRXVQmvLBsNMh8gMswPJ+CNUuawfftvk/vrRHavWo9Eg1yJdUdFDuixf6LdFJENUOeOeHIxvvKNx35yDB8V1zvxG8CRWzW4qSJTkPths7Rkq1fy8zDnVvNumF1SYGje+2Dn0NnyT1yRl2+cN0Vx0zG75yUUBza5oEKV721OQoQZa2BfRyf2HVRHuolRbsK5q3U3et5cpdHwV7QthOZgWN/aE="
    b64 = GMG_Grid
    GMG_Grid = GMG_Grid.replace('"', '')
    b64 = GMG_Grid.encode('utf-8') 
    buf = decodebytes(b64)

    buf1 = bytearray(b'\0'*len(buf))
    key = [ord('g'), ord('m'), ord('g'), ord('0')]
    for i in range(len(buf)):
        buf1[i] = buf[i] ^ key[i%4]

    print(zlib.decompress(buf1[4:]).decode())

    with open(filepath + ".xml", "w") as datei:
        # Den String in die Datei schreiben
        datei.write(zlib.decompress(buf1[4:]).decode())


    
    return zlib.decompress(buf1[4:]).decode()

def decodeGMGPrintProductConfiguration (filepath):
   

    try:
        file = open(filepath, "r")
        read = file.readlines()
        file.close()

        modified = []
        for line in read:
            modified.extend(line.split())

        for statement in modified:
            if statement == "GMG_PrintProductConfiguration":
                GMG_Grid = modified[modified.index(statement)+1]

    except FileNotFoundError:
        print("This filepath dose not exist")

  
    


    #b64 = b"Z2xDYR/xiq02Ar14cyu4Z72Sd8H5VUPrQ0ZeZEqwKpuqA/aG6rmgOWupPVPI4b4emNOHMq6Nfr9UR4ro+9Pj6JD9XMcZMn+ozFkbyZhZWcooC9QYKsNc7fS+qaEr644YLUuwrehyGQ/Zh9vcmpE0yA1VOU5eN5lrWhFM9S8LjDuiqNyfKJxJKcDeByN+geFepRe6AL9RO+9+5RvIF3aqwmjmGFUIUejT7vWpw1c9CJzEAtVYkyuD0o8sq94jUfDn+lH8+6m+87qfE/1VFfVIT7dnXk5KC7YAa/NbZXY3W+W6221/+MvfzxBDQULRHtKIlDZLl1WkUocNsTFyKa6HKbawNFvbmFn59nGuBUfUrUtYByofPbl45qU3PzsMjDBJ+p2sJ2C2GqYZ5SxftDdpiOpBmqA4iamuMG/IO8tRcKXFP7lnzSDIiBPwLWHOitjM4CyOLEJhMXdIhmxR11Sx6Tt1A9lf54LaFZJGveEfkUgtHsEskZm0CKop9d95vwtfc0o0igPyYYysiNPSe8WjzELL1ec45UM88yZxnF8T3SdUlO/d+yVO6Dxc3CWQ0Na4MKmdCySBleaWgC1LOHIzdO4g85jt17h0jhOFU532hd3CrRgeIv6Dsn2C1vb4CXuX+ESWLiyRXSakEs/AeUabTp4FRRmWcyxM9CSum7pTTExwyNb8TtzR9A3t3aOzAGaHcndLAlbgIZWKDKIekWLe6FEOkmh0P+su/jrcaNHTqU68ccW0sreUG6OoLo2QeG3kWlYWsMkjjY2uPU+3i6PcHgxX2EQ/vkjXSQbNx7EhIqydGYbiwGbADl5NMH4a1xXZtMIpTc2cQYfvEqkrgNB+KSiPri1tIlHh9NVsR855ieJAltrs4iKXuS2MdD2FiXQk6I9KUU9zipK6J89dvSjimLdhnpwlmcMV92u1AHdjtZCsGT67t10kaOx3Id0SpZCZGs7NeC4n43Mud9KOCPmLWpvFBQJAEUT3BXOAdTeCVTzzgqx8f4XBsCskM2TOdV/qqVdUQQPu/14Xmlqv1d4khEPmQMHDJ4b5QRd1R4ZD8vI6lwNNWXimF59p+OhGeejSSgxPLorPunrPAL114sW3PJ7BJcTDSPc2tV13fHu8Ah2lqXY9tE2zbuBdOWNFRS65B4/PZtUzKCzKkkstlCtRuL5rEOeUdFzLJCbVSHUizST5AcUXFnqJYk6xwn+xilQhG+8owO5wJcFPs3W2hbxJ6srpvZY6Oe86qv2oGiMiPTIOTWozJvJok6jZiyA9K5I9BtMxryWHd3q1gZItpY22yth/A8Cva1bUh4MdDgsK/FIueyjSLviloyqZQ96lME213JmuyhtV5X+XZFmBJEV59TyvNwy9KNylN7bTu208ximVNOLYLiZHPcMMR8gkKhNp7F2RcqRZb4WwTGvn82UtYkX1UJrywbDDIYJUoStJZHcR5YY2ELaUEVx2I3vVFXV/zXn5XMVKFieJP07swiKx1yR8G7Ione4tJZxP7nXrheFJt8q0vcs6ZO9nqqCoRyN/PW8OEGpupvB2MLESa3G2AGovikVvvlIV5f/zOGHjDRWMmKdhGEyfyyFYR3Z8sSr/IIELP0qkhXlbxDVI1eT59DrVLJUX+llp+OIjwG5TphcRZaKTHyZ/uiDWD4d1B/xGNkxNBCRTzCBaBG6U5llrNIQrE2nsXQl0pFlvhbBMq+bzZS1iRXVQmvLBsNMh8gMswPJ+CNUuawfftvk/vrRHavWo9Eg1yJdUdFDuixf6LdFJENUOeOeHIxvvKNx35yDB8V1zvxG8CRWzW4qSJTkPths7Rkq1fy8zDnVvNumF1SYGje+2Dn0NnyT1yRl2+cN0Vx0zG75yUUBza5oEKV721OQoQZa2BfRyf2HVRHuolRbsK5q3U3et5cpdHwV7QthOZgWN/aE="
    b64 = GMG_Grid
    GMG_Grid = GMG_Grid.replace('"', '')
    b64 = GMG_Grid.encode('utf-8') 
    buf = decodebytes(b64)

    buf1 = bytearray(b'\0'*len(buf))
    key = [ord('g'), ord('m'), ord('g'), ord('0')]
    for i in range(len(buf)):
        buf1[i] = buf[i] ^ key[i%4]

    print(zlib.decompress(buf1[4:]).decode())

    with open(filepath + "_PrintProductConfiguration.xml", "w") as datei:
        # Den String in die Datei schreiben
        datei.write(zlib.decompress(buf1[4:]).decode())


    
    return zlib.decompress(buf1[4:]).decode()



def createSpectralMeasurementData(input_file: str, output_file: str):

        #ICCprofilepath = "C:\\Users\\henning\\EpsonSC9000_sm250_7C-Inkjet_330_Large_2_i1v2-M0.txt_GCR0.icc"
        #ICCprofilepath = "C:\\Test_Data\\Rosa\\Testcharts_DoubleGrid\\C500_Pavo_Brown_Test V1.mxn.srcP.icc"
        ICCprofilepath = "C:\\Test_Data\\Rosa\\testfiles_Durst\\ISOcoatedv2-39L.icc"        


        # Delete old, existing output file
        if os.path.exists(output_file):
            os.remove(output_file)

        # Step 2: Initialize the output file
        if os.path.exists(output_file):
            os.remove(output_file)

        # Step 3: Find the line number of 'NUMBER_OF_FIELDS'
        with open(input_file, 'r') as file:
            lines = file.readlines()

        line_number = None
        for i, line in enumerate(lines, 1):
            if "NUMBER_OF_FIELDS" in line:
                line_number = i
                break

        # Step 4: Extract lines up to the line before 'NUMBER_OF_FIELDS'
        with open(output_file, 'w') as file:
            if line_number:
                file.writelines(lines[:line_number - 1])

        # Step 5: Run external commands
        subprocess.run(["copycgatsdata", "DEV", input_file, "1_text.txt"])

        with open("1_text.txt", "r") as file:
            input_data = file.read()

        result = subprocess.run(["transiccgmg.exe", "-i", ICCprofilepath, "-o", "*Lab", "-t", "3", "-c", "0", "-n"],
            input=input_data, text=True, stdout=open("2_Lab.txt", "w")  
        )


        #this works only for CMYK. In case of 5 colors, CMYK has to be replaced by "5CLR"
        subprocess.run(["copydatacgats", "+ID", "1_text.txt", "CMYK", "2_Lab.txt", "LAB", "3_Out.txt"])
        #subprocess.run(["copydatacgats", "+ID", "1_text.txt", "7CLR", "2_Lab.txt", "LAB", "3_Out.txt"])
        print("3_Out.txt created")
        subprocess.run(["lab2spectralApp", "-scg", "3_Out.txt", "4_spec.txt"])
        print("4_spec.txt created")

        # Step 6: Find the line number of 'NUMBER_OF_FIELDS' in spec.txt and subtract 2
        with open("4_spec.txt", 'r') as file:
            spec_lines = file.readlines()

        line_number2 = None
        for i, line in enumerate(spec_lines, 1):
            if "NUMBER_OF_FIELDS" in line:
                line_number2 = i
                break

        if line_number2:
            line_number2 -= 2

        # Step 7: Extract lines and append to outputFile
        with open(output_file, 'a') as file:
            if line_number2:
                file.writelines(spec_lines[line_number2:])

        # Step 8: Clean up temporary files
        for temp_file in ["2_Lab.txt", "3_Out.txt", "4_spec.txt", "1_text.txt", "Lab2SpecLog.txt"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)



        m = Measurement(output_file)
        #m.save(output_file + ".txt")
        #m.storeInternal(output_file + ".json")

        #This shall be removed.
        #m.set_print_product_configuration(PrintProductParams())
 
                           
        PrinterProfileParams = PrinterProfileCalculationParams()
        PrinterProfileParams.metaInfo.description = os.path.basename(output_file)
        PrinterProfileParams.metaInfo.copyright = "Copyright 2024 (C) GMG GmbH & Co. KG"

        PrinterProfileParams.targetColorSpace.maxTac = 330
        PrinterProfileParams.separationParams.gamutMapping.blackPointCompensation = 0
        PrinterProfileParams.separationParams.gamutMapping.expansionIntensity = 0
        PrinterProfileParams.separationParams.gamutMapping.grayAxisAlignment = 30
        PrinterProfileParams.separationParams.ucrLight = 30
        PrinterProfileParams.separationParams.ucrMidtone = 50
        PrinterProfileParams.separationParams.ucrShadow = 70
        PrinterProfileParams.separationParams.blackWidth = 100
        PrinterProfileParams.separationParams.maxBlackValue = 100
        PrinterProfileParams.separationParams.blackStart = 0
        p = Profile.calculate_printer_profile(m, PrinterProfileParams,True)             

        p.save_as_icc(output_file + ".icc")









def main():
    engine = RosaEngine()


 

 
    print('start')
 
    folder_path_input = 'C:\\Test_Data\\Profiles_Evaluation\\Input\\'
    folder_path_temp = 'C:\\Test_Data\\Profiles_Evaluation\\Temp\\'
    folder_path_output = 'C:\\Test_Data\\Profiles_Evaluation\\Output\\'

   
    if os.path.isdir(folder_path_input):
        for file_name in os.listdir(folder_path_input):
            if file_name.endswith('.icc'):
                inputfile = folder_path_input + file_name
                outputfile = folder_path_output + os.path.splitext(file_name)[0] + ".txt"
                dstp = folder_path_input + os.path.splitext(file_name)[0] + ".dstP"
                inpP = folder_path_input + os.path.splitext(file_name)[0] + ".inpP"
                print(inputfile)

                subprocess.run(["icc2tags", inputfile, "dstP"])
                subprocess.run(["icc2tags", inputfile, "inpP"] )

                with open(folder_path_input + "Input_Lab1.txt", "r") as file:
                    Input_Lab1 = file.read()

                with open(folder_path_input + "Input_Lab2.txt", "r") as file:
                    Input_Lab2 = file.read()        

                print("input Lab")
                print(Input_Lab1)
                print(Input_Lab2)            

                inputRGB1 = subprocess.run(["transiccgmg.exe", "-o", inpP, "-i", "*Lab", "-t", "3", "-c", "0", "-n"],
                    input=Input_Lab1, text=True, capture_output=True)

                inputRGB2 = subprocess.run(["transiccgmg.exe", "-o", inpP, "-i", "*Lab", "-t", "3", "-c", "0", "-n"],
                    input=Input_Lab2, text=True, capture_output=True)

                print("input RGB")
                print(inputRGB1.stdout)
                print(inputRGB2.stdout)

                outputCLR1 = subprocess.run(["transiccgmg.exe", "-l", inputfile, "-c", "0", "-n"],
                    input=str(inputRGB1.stdout), text=True, capture_output=True)

                outputCLR2 = subprocess.run(["transiccgmg.exe", "-l", inputfile, "-c", "0", "-n"],
                    input=str(inputRGB2.stdout), text=True, capture_output=True)

                print("output values")
                print(outputCLR1.stdout)
                print(outputCLR2.stdout)

                outputLab1 = subprocess.run(["transiccgmg.exe", "-i", dstp, "-o", "*Lab", "-t", "3", "-c", "0", "-n"],
                    input=outputCLR1.stdout, text=True, capture_output=True)

                outputLab2 = subprocess.run(["transiccgmg.exe", "-i", dstp, "-o", "*Lab", "-t", "3", "-c", "0", "-n"],
                    input=outputCLR2.stdout, text=True, capture_output=True)

                print("output Lab")
                print(outputLab1.stdout)
                print(outputLab2.stdout)

                Lab1_array = []
                Lab2_array = []
                Lab3_array = []
                Lab4_array = []


 
                # Replace commas with dots
                Input_Lab1 = Input_Lab1.replace(',', '.')

                # Split into lines and process each line
                lines = Input_Lab1.strip().split('\n')
                Lab1_array = np.array([[float(value) for value in line.split()] for line in lines])

                    
                Input_Lab2 = Input_Lab2.replace(',', '.')

                # Split into lines and process each line
                lines = Input_Lab2.strip().split('\n')
                Lab2_array = np.array([[float(value) for value in line.split()] for line in lines])

                outputLab1 = str(outputLab1.stdout.replace(',', '.'))

                # Split into lines and process each line
                lines = outputLab1.strip().split('\n')
                Lab3_array = np.array([[float(value) for value in line.split()] for line in lines])


                outputLab2 = str(outputLab2.stdout.replace(',', '.'))

                # Split into lines and process each line
                lines = outputLab2.strip().split('\n')
                Lab4_array = np.array([[float(value) for value in line.split()] for line in lines])



                combined_lab_array = np.hstack((Lab1_array, Lab2_array, Lab3_array, Lab4_array))
                print(combined_lab_array)
                print(combined_lab_array[0,0])
                #print(combined_lab_array[3,0])
                

                num_rows = combined_lab_array.shape[0]
                result = np.zeros((num_rows, 6))

                with open(outputfile, "w") as output_file:
                    for idx, row in enumerate(combined_lab_array):
                        #print("Row:", idx)

                        deltaE1 = math.sqrt((combined_lab_array[idx,0]-combined_lab_array[idx,3])**2 + (combined_lab_array[idx,1]-combined_lab_array[idx,4])**2 +(combined_lab_array[idx,2]-combined_lab_array[idx,5])**2)
                        deltaE2 = math.sqrt((combined_lab_array[idx,6]-combined_lab_array[idx,9])**2 + (combined_lab_array[idx,7]-combined_lab_array[idx,10])**2 +(combined_lab_array[idx,8]-combined_lab_array[idx,11])**2)
                        #print(deltaE1)
                        #print(deltaE2)
                        result[idx,0] = deltaE1
                        result[idx,1] = deltaE2
                        result[idx,2] = (deltaE2 / deltaE1 ) * 100

                        #DeltaL2/DeltaL1
                        deltaL1 = float(combined_lab_array[idx,0]-combined_lab_array[idx,3])
                        deltaL2 = float(combined_lab_array[idx,6]-combined_lab_array[idx,9])
                        result[idx,3] =  (deltaL2 / deltaL1 ) * 100
                        

                        #DeltaC2/DeltaC1
                        deltaC1 = math.sqrt(combined_lab_array[idx,1]**2+combined_lab_array[idx,2]**2) - math.sqrt(combined_lab_array[idx,4]**2+combined_lab_array[idx,5]**2)
                        deltaC2 = math.sqrt(combined_lab_array[idx,7]**2+combined_lab_array[idx,8]**2) - math.sqrt(combined_lab_array[idx,10]**2+combined_lab_array[idx,11]**2)
                        
                        #deltaC2 = math.sqrt((combined_lab_array[idx,7]-combined_lab_array[idx,10])**2 +(combined_lab_array[idx,8]-combined_lab_array[idx,11])**2)

                        deltaH1 =  0  
                        deltaH2 = 0

                        if deltaC1 !=0: 
                            result[idx,4] = (deltaC2 / deltaC1) * 100

                            deltaH1 =   (deltaE1**2 - deltaL1**2 - deltaC1**2)  
                            deltaH2 =   (deltaE2**2 - deltaL2**2 - deltaC2**2) 


                            
                        if deltaH1>0 and deltaH2>0:
                            deltaH1 = math.sqrt(deltaH1)
                            deltaH2 = math.sqrt(deltaH2)
                            result[idx,5] = (deltaH2 / deltaH1)* 100
                        

                        #print ("{:5.1f}".format(deltaE1), "{:5.1f}".format(deltaL1), "{:5.1f}".format(deltaC1), "{:5.1f}".format(deltaH1))
                        #print ("{:5.1f}".format(deltaE2), "{:5.1f}".format(deltaL2), "{:5.1f}".format(deltaC2), "{:5.1f}".format(deltaH2))


                        printstring = "L:" + "{:5.1f}".format(combined_lab_array[idx,0]) + " a:"+ "{:5.1f}".format(combined_lab_array[idx,1])+ " b:"+ "{:5.1f}".format(combined_lab_array[idx,2])+ " dE "+f'{(deltaE1):4.1f}'+" dC "+f'{(deltaC1):4.1f}'+" dH "+f'{(deltaH1):4.1f}'+"   Scaling  De76: "f'{(result[idx,2]):5.1f}'+ "%"+ "   DL: "f'{(result[idx,3]):5.1f}'+ "%"+ "   DC: "f'{(result[idx,4]):5.1f}'+ "%"+ "   DH: "f'{(result[idx,5]):5.1f}'+ "%\n"
                        print(printstring)
                        output_file.write(printstring)
                        #output_file.write(f"Inks: CMYK, Quality: {quality_name}, Number of Pages: {Num_of_pages}, Number of Sets: {num_of_sets}\n")



if __name__ == "__main__":
    main()
