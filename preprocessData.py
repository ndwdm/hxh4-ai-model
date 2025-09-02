#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 01 18:29:16 2025

@author: ndw
"""

import csv
import datetime
import pandas as pd
from math import degrees, radians, sin, floor, ceil, fabs
from typing import Optional

def julianday(date: datetime.date) -> float:
    """Calculate the Julian Day for the specified date"""
    y = date.year
    m = date.month
    d = date.day

    if m <= 2:
        y -= 1
        m += 12

    a = floor(y / 100)
    b = 2 - a + floor(a / 4)
    jd = floor(365.25 * (y + 4716)) + floor(30.6001 * (m + 1)) + d + b - 1524.5

    return jd

def proper_angle(value: float) -> float:
    if value > 0.0:
        value /= 360.0
        return (value - floor(value)) * 360.0
    else:
        tmp = ceil(fabs(value / 360.0))
        return value + tmp * 360.0


def _phase_asfloat(date: datetime.date) -> float:
    jd = julianday(date)

    DT = pow((jd - 2382148), 2) / (41048480 * 86400)

    T = (jd + DT - 2451545.0) / 36525
    T2 = pow(T, 2)
    T3 = pow(T, 3)
    D = 297.85 + (445267.1115 * T) - (0.0016300 * T2) + (T3 / 545868)
    D = radians(proper_angle(D))
    M = 357.53 + (35999.0503 * T)
    M = radians(proper_angle(M))
    M1 = 134.96 + (477198.8676 * T) + (0.0089970 * T2) + (T3 / 69699)
    M1 = radians(proper_angle(M1))
    elong = degrees(D) + 6.29 * sin(M1)
    elong -= 2.10 * sin(M)
    elong += 1.27 * sin(2 * D - M1)
    elong += 0.66 * sin(2 * D)
    elong = proper_angle(elong)
    elong = round(elong)
    moon = ((elong + 6.43) / 360) * 29.530588853#28
    
    return moon

def phase(date: Optional[datetime.date] = None) -> float:
    """Calculates the phase of the moon on the specified date.

    Args:
        date: The date to calculate the phase for. Dates are always in the UTC timezone.
              If not specified then today's date is used.

    Returns:
        A number designating the phase.

        ============  ==============
        0 .. 6.99     New moon
        7 .. 13.99    First quarter
        14 .. 20.99   Full moon
        21 .. 27.99   Last quarter
        ============  ==============
    """

    if date is None:
        date = datetime.date.today()

    moon = _phase_asfloat(date)
    if moon >= 28.0:
        moon -= 28.0
    return moon

with open('dataset/votingDataset.csv', 'r') as data, open('dataset/votingPreprocessed.csv', 'w') as out:
        reader = csv.reader(data)
        writer = csv.writer(out)
        
        headers = next(reader)
        headers.append("dayNumberOfYear")
        headers.append("moonPhase")
        headers.append("meanValue")
        headers.append("quality")

        writer.writerow(headers)
        for row in reversed(list(reader)):
            bad_count = float(row[3])
            mid_count = float(row[4])
            good_count = float(row[5])
            
            bad_coeff = bad_count * 1
            mid_coeff = mid_count * 2
            good_coeff = good_count * 3

            mean_value = (bad_coeff + mid_coeff + good_coeff) / (bad_count + mid_count + good_count)
            quality = 1
                
            #
            if mean_value >= 1 and mean_value < 2:
                quality = 0
            else:
                quality = 1
            #
    
            date = row[2]            
            period = pd.Period(date, freq='H')
            day_number_of_year = period.day_of_year

            convertedDate = datetime.datetime.strptime(date, "%Y-%m-%d").date()            
            moon_value = phase(convertedDate)
            moon_phase = 0
            
            if moon_value >= 0 and moon_value <= 6.99:
                moon_phase = 0
            elif moon_value >= 7.0 and moon_value <= 13.99:
                moon_phase = 1
            elif moon_value >= 14.0 and moon_value <= 20.99:
                moon_phase = 2
            elif moon_value >= 21.0 and moon_value <= 27.99:
                moon_phase = 3
                
            row.append(day_number_of_year) # Adding the field with the day number of the year
            row.append(moon_phase) # Adding the field with the moon phase for the current voted day
            row.append(mean_value) # Adding the field with the mean value for the current voted day
            row.append(quality) # Adding the field with the binary bad or good quality value for the voted day
                        
            writer.writerow(row)
