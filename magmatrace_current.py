#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:02:42 2020

@author: jordanlubbers / chucklewis
"""

"""magmatrace is a python module for doing high temperature geochemistry calculations

"""

# import the dependencies we'll need in these functions:
import numpy as np
import pandas as pd
import warnings

import re

#%% Mixing model related functions
def mixing(c1, c2, f):
    """
    mixing creates a mixing model between two endmembers

    Inputs:
        c1 = concentration of endmember 1
        c2 = concentration of endmember 2
        f = fraction of endmember 1 in the model

    Returns:
        cm = concnetration of the mixture
    """
    cm = c1 * f + c2 * (1 - f)
    return cm


# isotopic mixing model.
# fix this to add in a conditional to choose either one or two ratios
def isomix(rationum, c1, r1, c2, r2, *data):
    """
    isomix uses equations 18.24-18.26 from Faure 1998 to calculate isotopic mixing
    compositions for a given isotopic pair

    Inputs:
    rationum: use the input 'oneratio' or 'tworatios' to define how many isotopic 
            systems you are interested in
    c1 = concentration of element for endmember 1
    c2 = concentration of element for endmember 2
    r1 = isotopic ratio for endmember 1
    r2 = isotopic ratio for endmember 2
    *data = repeat the first 4 inputs for the second isotopic system of interest

    Returns:

    cm = concentrations of mixture for various values of 'f' where
    f is the fraction of endmember 1 in the mixture

    rm = isotopic ratios of mixture for various values of 'f'
    """

    # array of fractions of component 1
    f = np.linspace(0, 1, 11)
    # concentration of the mixture
    # eq. 18.19
    if rationum == "oneratio":
        cm = c1 * f + c2 * (1 - f)

        # eq. 18.25
        a = (c1 * c2 * (r2 - r1)) / (c1 - c2)
        # eq. 18.26
        b = (c1 * r1 - c2 * r2) / (c1 - c2)
        # eq. 18.24
        rm = a / cm + b

        return cm, rm

    elif rationum == "tworatios":
        cm = c1 * f + c2 * (1 - f)

        # eq. 18.25
        a = (c1 * c2 * (r2 - r1)) / (c1 - c2)
        # eq. 18.26
        b = (c1 * r1 - c2 * r2) / (c1 - c2)
        # eq. 18.24
        rm = a / cm + b

        cm2 = data[0] * f + data[2] * (1 - f)

        # eq 18.25
        c = (data[0] * data[2] * (data[3] - data[1])) / (data[0] - data[2])
        # eq 18.26
        d = (data[0] * data[1] - data[2] * data[3]) / (data[0] - data[2])
        rm2 = c / cm2 + d

        return cm, rm, cm2, rm2
    else:
        print(
            "Check your input. Ensure to specify rattionum and the correct amount of concentrations or ratios"
        )

def ratio_mixing(df, n_components, resolution=0.1):

    """
    Mixing of ratios as described by Albarede 1995
    Introduction to Geochemical Modeling equation 1.3.1
    
    Inputs:
    
    df | pandas DataFrame
    
    DataFrame of inputs. should be formatted as follows:
    
    For 2 component mixing:
    
    Index|Element1_c|Element1_r|Element2_c|Element2_r
    -------------------------------------------------
      A  |          |          |          |
    -------------------------------------------------  
      B  |          |          |          |
    
      
    
    For 3 component mixing:
    
    Index|Element1_c|Element1_r|Element2_c|Element2_r
    -------------------------------------------------
      A  |          |          |          |
    -------------------------------------------------  
      B  |          |          |          |
    -------------------------------------------------
      C  |          |          |          |
      
      
    Where the name of each component is the index of the dataframe and the 
    concentration and ratio columns for each elemental species contain "_c" and "_r" 
    somewhere in the column header, respectively. 
    
    n_components | int
    
    Number of end-member components (either 2 or 3)
    
    resolution | float
    
    The resolution you want to run your mixing model at. This is a number between 0.01 
    and 0.5. This is how far apart to space points in the eventual mixing mesh
    (e.g. .1 will return a mixing mesh spaced by 1O% increments for each component)
    
    Default is 0.1
    
    
    
    
    Returns:
    
    results | pandas DataFrame
    
    The results of the mixing model that is n x 7 in shape:
    
    f_A|f_B|f_C|Element1_c_mix|Element2_c_mix|Element1_r_mix|Element2_r_mix
    -----------------------------------------------------------------------
    
    Where f columns are fraction of each component in the mixture and other columns
    Are for the concentrations and ratios of the mixture for each respective combination
    of f values
    
    
    """

    if n_components == 2:

        if resolution < 0.01:
            print(
                "Please pick a lower resolution (e.g., bigger number).\nYou don't need it and it your computer may explode"
            )

        if resolution > 0.5:
            print("Please pick a higher resolution (e.g., number < 0.5). \n")

        else:

            # generate an array for fraction of each component
            f = np.arange(0, 1 + resolution, resolution)

            # all possible combinations for three f arrays
            a = np.array(np.meshgrid(f, f)).T.reshape(-1, 2)

            # where the combinations sum to 1
            f_vals = a[a.sum(axis=1) == 1]

            # get names of components
            components = df.index.tolist()

            # get names of columns where concentrations and ratios are held
            # IMPORTANT TO HAVE DATAFRAME IN THIS FORMAT
            elements = [col for col in df.columns if "_c" in col]
            ratios = [col for col in df.columns if "_r" in col]

            # Concentration of mixture

            if len(elements) == 1:

                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = df.loc[components[0], ratios[0]] * (
                    (f_vals[:, 0] * df.loc[components[0], elements[0]])
                    / el1_mix_concentrations
                ) + df.loc[components[1], ratios[0]] * (
                    (f_vals[:, 1] * df.loc[components[1], elements[0]])
                    / el1_mix_concentrations
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                    }
                )
            else:

                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                )
                el2_mix_concentrations = (
                    df.loc[components[0], elements[1]] * f_vals[:, 0]
                    + df.loc[components[1], elements[1]] * f_vals[:, 1]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = df.loc[components[0], ratios[0]] * (
                    (f_vals[:, 0] * df.loc[components[0], elements[0]])
                    / el1_mix_concentrations
                ) + df.loc[components[1], ratios[0]] * (
                    (f_vals[:, 1] * df.loc[components[1], elements[0]])
                    / el1_mix_concentrations
                )

                el2_mix_ratios = df.loc[components[0], ratios[1]] * (
                    (f_vals[:, 0] * df.loc[components[0], elements[1]])
                    / el2_mix_concentrations
                ) + df.loc[components[1], ratios[1]] * (
                    (f_vals[:, 1] * df.loc[components[1], elements[1]])
                    / el2_mix_concentrations
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(elements[1]): el2_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                        "{}_mix".format(ratios[1]): el2_mix_ratios,
                    }
                )

    if n_components == 3:

        if resolution < 0.01:
            print(
                "Please pick a lower resolution (e.g., bigger number).\nYou don't need it and it your computer may explode"
            )

        if resolution > 0.5:
            print("Please pick a higher resolution (e.g., number < 0.5). \n")

        else:

            # generate an array for fraction of each component
            f = np.arange(0, 1 + resolution, resolution)

            # all possible combinations for three f arrays
            a = np.array(np.meshgrid(f, f, f)).T.reshape(-1, 3)

            # where the combinations sum to 1
            f_vals = a[a.sum(axis=1) == 1]

            # get names of components
            components = df.index.tolist()

            # get names of columns where concentrations and ratios are held
            # IMPORTANT TO HAVE DATAFRAME IN THIS FORMAT
            elements = [col for col in df.columns if "_c" in col]
            ratios = [col for col in df.columns if "_r" in col]

            if len(elements) == 1:
                # Concentration of mixture using basic 3 component mixing
                # of concentrations
                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                    + df.loc[components[2], elements[0]] * f_vals[:, 2]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = (
                    df.loc[components[0], ratios[0]]
                    * (
                        (f_vals[:, 0] * df.loc[components[0], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[1], ratios[0]]
                    * (
                        (f_vals[:, 1] * df.loc[components[1], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[2], ratios[0]]
                    * (
                        (f_vals[:, 2] * df.loc[components[2], elements[0]])
                        / el1_mix_concentrations
                    )
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "f_{}".format(components[2]): f_vals[:, 2],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                    }
                )

            else:

                # Concentration of mixture using basic 3 component mixing
                # of concentrations
                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                    + df.loc[components[2], elements[0]] * f_vals[:, 2]
                )
                el2_mix_concentrations = (
                    df.loc[components[0], elements[1]] * f_vals[:, 0]
                    + df.loc[components[1], elements[1]] * f_vals[:, 1]
                    + df.loc[components[2], elements[1]] * f_vals[:, 2]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = (
                    df.loc[components[0], ratios[0]]
                    * (
                        (f_vals[:, 0] * df.loc[components[0], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[1], ratios[0]]
                    * (
                        (f_vals[:, 1] * df.loc[components[1], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[2], ratios[0]]
                    * (
                        (f_vals[:, 2] * df.loc[components[2], elements[0]])
                        / el1_mix_concentrations
                    )
                )

                el2_mix_ratios = (
                    df.loc[components[0], ratios[1]]
                    * (
                        (f_vals[:, 0] * df.loc[components[0], elements[1]])
                        / el2_mix_concentrations
                    )
                    + df.loc[components[1], ratios[1]]
                    * (
                        (f_vals[:, 1] * df.loc[components[1], elements[1]])
                        / el2_mix_concentrations
                    )
                    + df.loc[components[2], ratios[1]]
                    * (
                        (f_vals[:, 2] * df.loc[components[2], elements[1]])
                        / el2_mix_concentrations
                    )
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "f_{}".format(components[2]): f_vals[:, 2],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(elements[1]): el2_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                        "{}_mix".format(ratios[1]): el2_mix_ratios,
                    }
                )

    return results
def isoassim(modeltype, rationum, r, D, cp, ca, ep, ea, *data):
    """
    
    isoassim uses equation 15B from DePaolo (1981) in order to look at
    the evolution of one or two isotopic ratios and their associated trace element
    concentrations during combined assimilation and fractionation
    
    Inputs:
    modeltype == DePaolo15b; this is the only one currently and is by far the most useful
    rationum = the number of isotopic systems you are interested in. List 'oneratio' or 'tworatios'
    r = the r value, defined as the rate of fractionation to the rate of assimilation by mass
    D = the bulk D value for the first isotopic system
    cp = concentration of trace element in parental magma
    ca = concentration of trace element in assimilant
    ep = isotopic ratio of parental magma
    ea = isotopic ratio of assimilant
    *data = if you are interested in two ratios as opposed to one then you must input new values
            for D through ea for the second isotopic system
    """
    # array of fractions of component 1
    f = np.linspace(0, 1, 11)

    if modeltype == "DePaolo15b" and rationum == "oneratio":
        # mix the trace elements
        cm = cp * f + ca * (1 - f)

        # get the effective distribution coefficient
        z = (r + D - 1) / (r - 1)

        # calculate the isotopic ratio of the daughter that has undergone assimilation
        em = ((r / (r - 1)) * (ca / z) * (1 - f ** (-z)) * ea + cp * f ** (-z) * ep) / (
            (r / (r - 1)) * (ca / z) * (1 - f ** (-z)) + (cp * f ** (-z))
        )

        return cm, em

    elif modeltype == "DePaolo15b" and rationum == "tworatios":

        # get mixes of both trace elements associated with the isotopic systems of interest
        cm = cp * f + ca * (1 - f)
        cm2 = data[1] * f + data[2] * (1 - f)

        # get the effective distribution coefficents for both isotopic systems
        z1 = (r + D - 1) / (r - 1)
        z2 = (r + data[0] - 1) / (r - 1)

        # calculate the isotopic ratios of the daughter for both systems
        em = ((r / (r - 1)) * (ca / z) * (1 - f ** (-z)) * ea + cp * f ** (-z) * ep) / (
            (r / (r - 1)) * (ca / z) * (1 - f ** (-z)) + (cp * f ** (-z))
        )
        em2 = (
            (r / (r - 1)) * (data[2] / z2) * (1 - f ** (-z2)) * data[4]
            + data[1] * f ** (-z2) * data[3]
        ) / ((r / (r - 1)) * (data[2] / z1) * (1 - f ** (-z1)) + (data[1] * f ** (-z1)))

        return cm, cm2, em, em2
    else:
        print(
            "You must specify the modeltype as DePaolo15b, number of ratios as one or two, r, D, and/or D2, then your ratios"
        )


# Equations by Aitcheson & Forrest (1994) used to estimate the degree of assimilation independent of
# Depaolo's (1981) variable r
# equations based on isotopic compositions
def crustfraciso(eq, systems, D, c0m, e0m, em, ea, *data):
    """
    This model will give either equation 5 or 7 of the Aitcheson & Forrest (1994) equations that are used for estimating
    the fraction of assimilated crust without the requirement of guessing at the r value required for the DePaolo (1981)
    equations. The user should be familiar about what needs to be input - in short, this is estimated basement compositions
    as the assimilant, measured compositions for the 'daughter', and a thoroughly educated guess at intital magma
    composition. An example of this applicability can be seen in Kay et al. (2010).
    
    Inputs:
    eq: 'five' for equation five and 'seven' for equation 'seven'.
        Equation five is independent of erupted magma composition and degree of melting F
        Equation seven is independent of the trace element composition in the assimilant
        
    systems: Up to four isotopic systems can be considered. These are designated by the input 'one', 'two', 'threee', or 
    'four'. There is a caveat to putting in more than one isotopic system explained by teh input parameter *data
    seen below
    
    D: The bulk partition coefficient of the element associated with the isotopic system of interest in the host magma
    
    c0m: Estimated trace element composition of the element associated with the isotopic system of interest in the 
    original parent magma.
    
    e0m: Estimated isotopic ratio for the system of interest in the original parent magma.
    
    em: Measured isotpic ratio of hte daughter magma that has undergone assimilation.
    
    ea: Estimated isotopic ratio of the assimilant.
    
    *data: If you wish to do more than one isotpic system, you must input values for D thorugh ea in exactly the same
    order as defined in the function above
    
    
    Outputs:
    crustfrac 1, 2, 3, or 4 depending on how many isotpic systems you are interested in. This is equivalent to the 
    value 'rho' in Aitcheson & Forrest (1994)
    """

    import numpy as np

    r = np.linspace(0, 1, 11)

    if eq == "five" and systems == "one":
        wave = (e0m - em) / (em - ea)
        ca = data[0]
        gamma = ca / c0m
        crustfrac = (r / (r - 1)) * (
            (1 + ((wave * (r + D - 1)) / (r * gamma))) ** ((r - 1) / (r + D - 1)) - 1
        )
        return crustfrac
    elif eq == "five" and systems == "two":
        wave1 = (e0m - em) / (em - ea)
        ca1 = data[0]
        gamma1 = ca / c0m
        crustfrac1 = (r / (r - 1)) * (
            (1 + ((wave1 * (r + D - 1)) / (r * gamma1))) ** ((r - 1) / (r + D - 1)) - 1
        )

        wave2 = (data[3] - data[4]) / (data[4] - data[5])
        ca2 = data[6]
        gamma2 = ca2 / data[2]
        crustfrac2 = (r / (r - 1)) * (
            (1 + ((wave2 * (r + data[1] - 1)) / (r * gamma2)))
            ** ((r - 1) / (r + data[1] - 1))
            - 1
        )
        return crustfrac1, crustfrac2
    elif eq == "five" and systems == "three":
        wave1 = (e0m - em) / (em - ea)
        ca1 = data[0]
        gamma1 = ca / c0m
        crustfrac1 = (r / (r - 1)) * (
            (1 + ((wave1 * (r + D - 1)) / (r * gamma1))) ** ((r - 1) / (r + D - 1)) - 1
        )

        wave2 = (data[3] - data[4]) / (data[4] - data[5])
        ca2 = data[6]
        gamma2 = ca2 / data[2]
        crustfrac2 = (r / (r - 1)) * (
            (1 + ((wave2 * (r + data[1] - 1)) / (r * gamma2)))
            ** ((r - 1) / (r + data[1] - 1))
            - 1
        )

        wave3 = (data[9] - data[10]) / (data[10] - data[11])
        ca3 = data[12]
        gamma3 = ca3 / data[8]
        crustfrac3 = (r / (r - 1)) * (
            (1 + ((wave3 * (r + data[7] - 1)) / (r * gamma3)))
            ** ((r - 1) / (r + data[7] - 1))
            - 1
        )
        return crustfrac1, crustfrac2, crustfrac3

    elif eq == "five" and systems == "four":
        wave1 = (e0m - em) / (em - ea)
        ca1 = data[0]
        gamma1 = ca / c0m
        crustfrac1 = (r / (r - 1)) * (
            (1 + ((wave1 * (r + D - 1)) / (r * gamma1))) ** ((r - 1) / (r + D - 1)) - 1
        )

        wave2 = (data[3] - data[4]) / (data[4] - data[5])
        ca2 = data[6]
        gamma2 = ca2 / data[2]
        crustfrac2 = (r / (r - 1)) * (
            (1 + ((wave2 * (r + data[1] - 1)) / (r * gamma2)))
            ** ((r - 1) / (r + data[1] - 1))
            - 1
        )

        wave3 = (data[9] - data[10]) / (data[10] - data[11])
        ca3 = data[12]
        gamma3 = ca3 / data[8]
        crustfrac3 = (r / (r - 1)) * (
            (1 + ((wave3 * (r + data[7] - 1)) / (r * gamma3)))
            ** ((r - 1) / (r + data[7] - 1))
            - 1
        )

        wave4 = (data[15] - data[16]) / (data[16] - data[17])
        ca4 = data[18]
        gamma4 = ca4 / data[14]
        crustfrac4 = (r / (r - 1)) * (
            (1 + ((wave4 * (r + data[13] - 1)) / (r * gamma4)))
            ** ((r - 1) / (r + data[13] - 1))
            - 1
        )
        return crustfrac1, crustfrac2, crustfrac3, crustfrac4

    elif eq == "seven" and systems == "one":
        cm = data[0]
        crustfrac = (r / (r - 1)) * (
            ((c0m / cm) * ((ea - e0m) / (ea - em))) ** ((r - 1) / (r + D - 1)) - 1
        )
        return crustfrac
    elif eq == "seven" and systems == "two":
        cm1 = data[0]
        crustfrac1 = (r / (r - 1)) * (
            ((c0m / cm1) * ((ea - e0m) / (ea - em))) ** ((r - 1) / (r + D - 1)) - 1
        )

        cm2 = data[6]
        crustfrac2 = (r / (r - 1)) * (
            ((data[2] / cm1) * ((data[5] - data[3]) / (data[5] - data[4])))
            ** ((r - 1) / (r + data[1] - 1))
            - 1
        )
        return crustfrac1, crustfrac2

    elif eq == "seven" and systems == "three":
        cm1 = data[0]
        crustfrac1 = (r / (r - 1)) * (
            ((c0m / cm1) * ((ea - e0m) / (ea - em))) ** ((r - 1) / (r + D - 1)) - 1
        )
        cm2 = data[6]
        crustfrac2 = (r / (r - 1)) * (
            ((data[2] / cm2) * ((data[5] - data[3]) / (data[5] - data[4])))
            ** ((r - 1) / (r + data[1] - 1))
            - 1
        )
        cm3 = data[12]
        crustfrac3 = (r / (r - 1)) * (
            ((data[8] / cm3) * ((data[11] - data[9]) / (data[11] - data[10])))
            ** ((r - 1) / (r + data[7] - 1))
            - 1
        )
        cm4 = data[18]
        crustfrac4 = (r / (r - 1)) * (
            ((data[14] / cm4) * ((data[17] - data[15]) / (data[17] - data[16])))
            ** ((r - 1) / (r + data[13] - 1))
            - 1
        )
        return crustfrac1, crustfrac2, crustfrac3, crustfrac4
    else:
        print("Check your input")


# equations independent of the isotopic composition
def crustfracele(systems, D, c0m, cm, ca, *data):
    """
    This model will give either equation 6of the Aitcheson & Forrest (1994) equations that are used for estimating
    the fraction of assimilated crust without the requirement of guessing at the r value required for the DePaolo (1981)
    equations. The user should be familiar about what needs to be input - in short, this is estimated basement compositions
    as the assimilant, measured compositions for the 'daughter', and a thoroughly educated guess at intital magma
    composition. An example of this applicability can be seen in Kay et al. (2010). This particular equation uses trace
    elements only and is independent of isotopic ratios. This equation is best used in combination with the function
    crustfraciso.
    
    Inputs:       
    systems: Up to four systems can be considered. These are designated by the input 'one', 'two', 'threee', or 
    'four'. There is a caveat to putting in more than one isotopic system explained by teh input parameter *data
    seen below
    
    D: The bulk partition coefficient of the element associated with the isotopic system of interest in the host magma
    
    c0m: Estimated trace element composition of the element associated with the isotopic system of interest in the 
    original parent magma.
    
    cm: Measured trace element composition of the 'daughter' magma that has undergone assimilation
    
    ca: Estimated trace element composition of the assimilant
    
    *data: If you wish to do more than one isotpic system, you must input values for D thorugh ea in exactly the same
    order as defined in the function above
    
    
    Outputs:
    crustfrac 1, 2, 3, or 4 depending on how many systems you are interested in. This is equivalent to the 
    value 'rho' in Aitcheson & Forrest (1994)
    """

    import numpy as np

    r = np.linspace(0, 1, 11)

    if systems == "one":
        crustfrac = (r / (r - 1)) * (
            ((c0m * (r + D - 1) - r * ca) / (cm * (r + D - 1) - r * ca))
            ** ((r - 1) / (r + D - 1))
            - 1
        )
        return (crustfrac,)
    elif systems == "two":
        crustfrac1 = (r / (r - 1)) * (
            ((c0m * (r + D - 1) - r * ca) / (cm * (r + D - 1) - r * ca))
            ** ((r - 1) / (r + D - 1))
            - 1
        )
        crustfrac2 = (r / (r - 1)) * (
            (
                (data[1] * (r + data[0] - 1) - r * data[3])
                / (data[2] * (r + data[0] - 1) - r * data[3])
            )
            ** ((r - 1) / (r + data[0] - 1))
            - 1
        )
        return crustfrac1, crustfrac2
    elif systems == "three":
        crustfrac1 = (r / (r - 1)) * (
            ((c0m * (r + D - 1) - r * ca) / (cm * (r + D - 1) - r * ca))
            ** ((r - 1) / (r + D - 1))
            - 1
        )
        crustfrac2 = (r / (r - 1)) * (
            (
                (data[1] * (r + data[0] - 1) - r * data[3])
                / (data[2] * (r + data[0] - 1) - r * data[3])
            )
            ** ((r - 1) / (r + data[0] - 1))
            - 1
        )
        crustfrac3 = (r / (r - 1)) * (
            (
                (data[5] * (r + data[4] - 1) - r * data[7])
                / (data[6] * (r + data[4] - 1) - r * data[7])
            )
            ** ((r - 1) / (r + data[4] - 1))
            - 1
        )
        return crustfrac1, crustfrac2, crustfrac3
    elif systems == "four":
        crustfrac1 = (r / (r - 1)) * (
            ((c0m * (r + D - 1) - r * ca) / (cm * (r + D - 1) - r * ca))
            ** ((r - 1) / (r + D - 1))
            - 1
        )
        crustfrac2 = (r / (r - 1)) * (
            (
                (data[1] * (r + data[0] - 1) - r * data[3])
                / (data[2] * (r + data[0] - 1) - r * data[3])
            )
            ** ((r - 1) / (r + data[0] - 1))
            - 1
        )
        crustfrac3 = (r / (r - 1)) * (
            (
                (data[5] * (r + data[4] - 1) - r * data[7])
                / (data[6] * (r + data[4] - 1) - r * data[7])
            )
            ** ((r - 1) / (r + data[4] - 1))
            - 1
        )
        crustfrac4 = (r / (r - 1)) * (
            (
                (data[9] * (r + data[8] - 1) - r * data[11])
                / (data[10] * (r + data[8] - 1) - r * data[11])
            )
            ** ((r - 1) / (r + data[8] - 1))
            - 1
        )
        return crustfrac1, crustfrac2, crustfrac3, crustfrac3
    else:
        print("Check your input")


#%% Thermometry related functions
def plag_kd_calc(element, An, temp, method):
    """
    calculates the partition coefficient for a given element in plagioclase based on its anorthite
    content according to the Arrhenius relationship as originally defined by Blundy and Wood (1991)
    
    This function gives the user an option of three experimental papers to choose from when calculating 
    partition coefficient:
    
    Bindeman et al., 1998 = ['Li','Be','B','F','Na','Mg','Al','Si','P','Cl','K','Ca','Sc',
    'Ti','Cr','Fe','Co','Rb','Sr','Zr','Ba','Y','La','Ce','Pr','Nd','Sm','Eu','Pb']
    
    Nielsen et al., 2017 = ['Mg','Ti','Sr','Y','Zr','Ba','La','Ce','Pr','Nd','Pb']
    
    Tepley et al., 2010 = ['Sr','Rb','Ba','Pb','La','Nd','Sm','Zr','Th','Ti']
    
    
    Inputs:
    -------
    element : string
    The element you are trying to calculate the partition coefficient for. See Bindeman 1998 for supported
    elements
    
    An : array-like
    Anorthite content (between 0 and 1) of the plagioclase. This can be a scalar value or Numpy array
    
    temp: scalar
    Temperature in Kelvin to calculate the partition coefficient at 
    
    method : string
    choice of 'Bindeman', 'Nielsen', 'Tepley'. This uses then uses the Arrhenius parameters from 
    Bindeman et al., 1998, Nielsen et al., 2017, or Tepley et al., 2010, respectively.
    
    Returns:
    --------
    kd_mean : array-like
    the mean partition coefficient for the inputs listed
    
    kd_std : array-like
    standard deviation of the partition coefficient calculated via 
    Monte Carlo simulation of 1000 normally distributed random A and B
    parameters based on their mean and uncertainties 
    
    """

    if method == "Bindeman":
        # Table 4 from Bindeman et al 1998
        elements = [
            "Li",
            "Be",
            "B",
            "F",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "Cl",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "Cr",
            "Fe",
            "Co",
            "Rb",
            "Sr",
            "Zr",
            "Ba",
            "Y",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Sm",
            "Eu",
            "Pb",
        ]

        a = (
            np.array(
                [
                    -6.9,
                    28.2,
                    -0.61,
                    -37.8,
                    -9.4,
                    -26.1,
                    -0.3,
                    -2,
                    -30.7,
                    -24.5,
                    -25.5,
                    -15.2,
                    -94.2,
                    -28.9,
                    -44,
                    -35.2,
                    -59.9,
                    -40,
                    -30.4,
                    -90.4,
                    -55,
                    -48.1,
                    -10.8,
                    -17.5,
                    -22.5,
                    -19.9,
                    -25.7,
                    -15.1,
                    -60.5,
                ]
            )
            * 1e3
        )
        a_unc = (
            np.array(
                [
                    1.9,
                    6.1,
                    0.5,
                    11.5,
                    1,
                    1.1,
                    0.8,
                    0.2,
                    4.6,
                    9.5,
                    1.2,
                    0.6,
                    28.3,
                    1.5,
                    6.3,
                    1.9,
                    10.8,
                    6.7,
                    1.1,
                    5.5,
                    2.4,
                    3.7,
                    2.6,
                    2.3,
                    4.1,
                    3.6,
                    6.3,
                    16.1,
                    11.8,
                ]
            )
            * 1e3
        )

        b = (
            np.array(
                [
                    -12.1,
                    -29.5,
                    9.9,
                    23.6,
                    2.1,
                    -25.7,
                    5.7,
                    -0.04,
                    -12.1,
                    11,
                    -10.2,
                    17.9,
                    37.4,
                    -15.4,
                    -9.3,
                    4.5,
                    12.2,
                    -15.1,
                    28.5,
                    -15.3,
                    19.1,
                    -3.4,
                    -12.4,
                    -12.4,
                    -9.3,
                    -9.4,
                    -7.7,
                    -14.2,
                    25.3,
                ]
            )
            * 1e3
        )
        b_unc = (
            np.array(
                [
                    1,
                    4.1,
                    3.8,
                    7.1,
                    0.5,
                    0.7,
                    0.4,
                    0.08,
                    2.9,
                    5.3,
                    0.7,
                    0.3,
                    18.4,
                    1,
                    4.1,
                    1.1,
                    7,
                    3.8,
                    0.7,
                    3.6,
                    1.3,
                    1.9,
                    1.8,
                    1.4,
                    2.7,
                    2.0,
                    3.9,
                    11.3,
                    7.8,
                ]
            )
            * 1e3
        )

        plag_kd_params = pd.DataFrame(
            [a, a_unc, b, b_unc], columns=elements, index=["a", "a_unc", "b", "b_unc"]
        )

        R = 8.314

    elif method == "Nielsen":
        elements = ["Mg", "Ti", "Sr", "Y", "Zr", "Ba", "La", "Ce", "Pr", "Nd", "Pb"]
        a = (
            np.array([-10, -32.5, -25, -65.7, -25, -35.1, -32, -33.6, -29, -31, -50])
            * 1e3
        )
        a_unc = np.array([3.3, 1.5, 1.1, 3.7, 5.5, 4.5, 2.9, 2.3, 4.1, 3.6, 11.8]) * 1e3

        b = np.array([-35, -15.1, 25.5, 2.2, -50, 10, -5, -6.8, 8.7, -8.9, 22.3]) * 1e3
        b_unc = np.array([2.1, 1, 0.7, 1.9, 3.6, 2.4, 2.3, 1.4, 2.7, 2.0, 7.8]) * 1e3

        plag_kd_params = pd.DataFrame(
            [a, a_unc, b, b_unc], columns=elements, index=["a", "a_unc", "b", "b_unc"]
        )

        R = 8.314

    elif method == "Tepley":
        elements = ["Sr", "Rb", "Ba", "Pb", "La", "Nd", "Sm", "Zr", "Th", "Ti"]
        a = (
            np.array(
                [-50.18, -35.7, -78.6, -13.2, -93.7, -84.3, -108.0, -70.9, -58.1, -30.9]
            )
            * 1e3
        )
        a_unc = (
            np.array([6.88, 13.8, 16.1, 44.4, 12.2, 8.1, 17.54, 58.2, 35.5, 8.6]) * 1e3
        )

        b = np.array(
            [44453, -20871, 41618, -15761, 37900, 24365, 35372 - 7042, -60465, -14204]
        )
        b_unc = np.array([1303, 2437, 2964, 5484, 2319, 1492, 3106, 101886073493])

        plag_kd_params = pd.DataFrame(
            [a, a_unc, b, b_unc], columns=elements, index=["a", "a_unc", "b", "b_unc"]
        )

        if np.percentile(An, q=50) < 0.6:
            warnings.warn(
                "Over half your An values are significantly below the calibration range in Tepley et al., (2010)"
                "and most likely will produce partition coefficient values that are significantly overestimated",
                stacklevel=2,
            )

        R = 8.314

    if element in elements:

        a = np.random.normal(
            plag_kd_params[element].a, plag_kd_params[element].a_unc, 1000
        )
        b = np.random.normal(
            plag_kd_params[element].b, plag_kd_params[element].b_unc, 1000
        )

        kds = np.exp((a[:, np.newaxis] * An + b[:, np.newaxis]) / (R * temp))

        kd_mean = np.mean(kds, axis=0)
        kd_std = np.std(kds, axis=0)

    else:
        raise Exception(
            "The element you have selected is not supported by this function. Please choose another one"
        )

    return kd_mean, kd_std

def amp_kd_calc(amph_sites_ff, element):
    """
    aem_calc calculates the partition coefficient for a specified trace element
    that is in equilibrium with a given amphibole composition according to
    Humphreys et al., 2019.
	
	supported elements = ['Rb','Sr','Pb','Zr','Nb','La','Ce','Nd','Sm',
	'Eu','Gd','Dy','Ho','Yb','Lu','Y']
    

    Parameters
    ----------
    amph_sites_ff : pandas DataFrame
        Amphibole site allocations that incorporate ferric ferrous iron.
        This should be the output from the get_amp_sites_ferric_ferrous function
    element : string
        The element you want to calculate the partition coefficient for 

    Raises
    ------
    Exception
        If you do not choose a supported element from Humphreys et al., 2019
        an error will be thrown prompting you to choose a supported element

    Returns
    -------
    aem_kd : array-like
        partition coefficient between amphibole and its equilibrium melt 
    aem_kd_se : scalar
        the one sigma uncertainty on your partition coefficient taken from 
        table 2 in Humphreys et al., 2019

    """

    # Building table 2 from Humphreys et al 2019
    elements = [
        "Rb",
        "Sr",
        "Pb",
        "Zr",
        "Nb",
        "La",
        "Ce",
        "Nd",
        "Sm",
        "Eu",
        "Gd",
        "Dy",
        "Ho",
        "Yb",
        "Lu",
        "Y",
    ]
    constants = np.array(
        [
            9.1868,
            3.41585,
            -4.2533,
            -25.6167,
            -22.27,
            -20.0493,
            -21.1078,
            -20.3082,
            -11.3625,
            -35.6604,
            -19.0583,
            -16.0687,
            -20.4148,
            -15.8659,
            -19.3462,
            -36.2514,
        ]
    )
    si = np.array(
        [
            -1.3898,
            -0.75281,
            0,
            2.6183,
            2.3241,
            2.0732,
            2.4749,
            2.5162,
            1.6002,
            4.1452,
            2.4417,
            2.3858,
            2.3654,
            2.281,
            2.1142,
            3.6078,
        ]
    )
    al = np.array([0, 0, 2.715, 2.6867, 0, 0, 0, 0, 0, 2.6886, 0, 0, 0, 0, 0, 3.78])
    ti = np.array(
        [
            -3.6797,
            0,
            1.69,
            4.838,
            3.7633,
            2.5498,
            2.4717,
            2.5863,
            0,
            6.4057,
            1.9786,
            1.8255,
            2.484,
            1.5905,
            2.8478,
            7.513,
        ]
    )
    fe3 = np.array(
        [
            -1.5769,
            0,
            0.7065,
            2.6591,
            2.9786,
            1.5317,
            1.5722,
            1.9459,
            1.2898,
            3.8508,
            1.8765,
            1.9741,
            3.2601,
            2.1534,
            2.7011,
            4.8366,
        ]
    )
    fe2 = np.array(
        [
            -0.6938,
            0.36529,
            0,
            0.6536,
            1.44,
            1.117,
            0.952,
            0.9566,
            1.2376,
            0.7255,
            0.9943,
            0.6922,
            1.2922,
            0.7867,
            1.0402,
            0.814,
        ]
    )
    ca = np.array(
        [
            0,
            0,
            0,
            2.5248,
            1.8719,
            2.2771,
            1.5311,
            1.2763,
            0,
            3.0679,
            1.3577,
            0,
            3.1762,
            0,
            2.9625,
            4.60,
        ]
    )
    naa = np.array(
        [0, 0, -1.0433, 0, 0, -1.4576, 0, 0, 0, 0, 0, 0, -4.9224, 0, -3.2356, 0]
    )
    se = np.array(
        [
            0.29,
            0.19,
            0.23,
            0.49,
            0.45,
            0.34,
            0.32,
            0.36,
            0.43,
            0.37,
            0.4,
            0.33,
            0.4,
            0.43,
            0.39,
            0.32,
        ]
    )
    columns = [
        "element",
        "constant",
        "Si",
        "Al_vi",
        "Ti",
        "Fe3",
        "Fe2",
        "Ca",
        "Na_A",
        "se",
    ]
    aem_params = pd.DataFrame(
        dict(
            constant=constants,
            Si=si,
            Al_vi=al,
            Ti=ti,
            Fe3=fe3,
            Fe2=fe2,
            Ca=ca,
            Na_a=naa,
            SE=se,
        ),
        index=elements,
    )

    if element in elements:

        aem_kd = np.exp(
            aem_params.loc[element].constant
            + (aem_params.loc[element].Si * amph_sites_ff["Si_T"])
            + (aem_params.loc[element].Al_vi * amph_sites_ff["Al_D"])
            + (aem_params.loc[element].Ti * amph_sites_ff["Ti_D"])
            + (aem_params.loc[element].Fe3 * amph_sites_ff["Fe3_D"])
            + (
                aem_params.loc[element].Fe2
                * (amph_sites_ff["Fe2_C"] + amph_sites_ff["Fe2_B"])
            )
            + (aem_params.loc[element].Ca * amph_sites_ff["Ca_B"])
            + (aem_params.loc[element].Na_a * amph_sites_ff["Na_A"])
        )
        aem_kd_se = aem_params.loc[element].SE
    else:
        raise Exception(
            "The element you have selected is not supported by this function. Please choose another one"
        )

    return aem_kd, aem_kd_se


# function to calculate zr saturation temperature
def t_zr_sat(M, zrmelt, model):
    """
    t_zr_sat calculates the zircon saturation temperature using
    the relationships found in both Watson and Harrison 1983 as
    well as Boehnke et al., 2013

        Inputs:
        M = (Na + K + 2Ca)/(Al*Si) in normalized cation fraction
        
        zrmelt = concentration of Zr in the melt
        
        model = 'watson' or 'boehnke'. This will govern the equation used
        to calculate zircon saturation temperature based on the equations 
        from watson and harrison 1983 or boehnke et al., 2013, respectively

        Returns:
        t = zircon saturation temperature for the chosen model

    BOTH TEMPERATURES ARE IN DEGREES CELSIUS
    
    """

    if model == "watson":

        t = 12900 / (2.95 + 0.85 * M + np.log(496000 / zrmelt)) - 273.15

    elif model == "boehnke":

        t = 10108 / (0.32 + 1.16 * M + np.log(496000 / zrmelt)) - 273.15

    return t


# titanium in quartz thermometry
def titaniq(Ti, P):
    """
    titaniq calculates the quartz crystallization temperature based on a known
    concentration of titanium in quartz and pressure of crystallization. This is
    based on the work of Huang and Audetát (2012).
    
        Inputs:
        Ti = array-like concentration of Ti in quartz (ppm)
        P = array-like pressure of crystallization (kbar)

        Returns:
        temp = array-like temperature of quartz crystallization (C)
    """
    temp = ((-2794.3 - (660.53 * P ** 0.35)) / (np.log10(Ti) - 5.6459)) - 273.15
    return temp


#%% melting and crystallization related functions

# partition coefficient


def kd(Cs, Cl):
    """
    kd calculates a partition coefficient for a given set of measurements. For 
    igneous petrology, this is commonly the concentration of a trace element in
    the mineral divided by the concentration of the same trace element in the
    melt (e.g. Rollinson 1993 Eq. 4.3)
    
        Inputs:
        Cs = concnetration in the mineral
        Cl = concentration in the melt
        
        Returns:
        kd = partition coefficient for the given input parameters
        
    """
    kd = Cs / Cl
    return kd


# Distribution coefficient
def bulk_kd(kds, f_s):
    """
    bulk_kd generates a distribution coefficient that is the weighted sum of 
    partition coefficients for an element in a given mineral assemblage.
    Based off Rollinson 1993 Eq. 4.5

    Parameters
    ----------
    kds : array-like
        the individual partition coefficients of the mineral assemblage
    f_s : array-like
        the individual fractions of each mineral in the overall assemblage
        between 0 and 1

    Returns
    -------
    bulk_kd : the bulk partition coefficient for a given trace element for
    the mineral assemblage

    """
    D = np.sum(kds * f_s)
    return D


# melting equations
def non_modal_batch_melt(Co, Do, F, P):
    """
    non_modal_batch calculates the concentration of a given trace element in a melt produced from non modal
    batch melting of a source rock as described by Shaw (1970) equation 15.
        Inputs:
        Co = Concentration of trace element in the original solid
        Do = Bulk distribution coefficient for element when F = 0
        F = Fraction of original solid melted (fraction of melt)
        P = Bulk distribution coefficient of the melting mineral assemblage

        Returns:
        Cl = concentration in the newly formed liquid

    Note: if Do and P are the same, then you effectively have modal batch melting
    """

    Cl = Co * (1 / (F * (1 - P) + Do))
    return Cl


def non_modal_frac_melt(Co, Do, F, P):
    """
    non_modal_frac_melt calculates the composition of a trace element in a melt produced from non modal
    fractional melting of a source rock as described by Rollinson 1993 Eq. 4.13 and 4.14.
        Inputs:
        Co = Concentration of trace element in the original solid
        Do = Bulk distribution coefficient for element when F = 0
        F = Fraction of original solid melted (fraction of melt)
        P = Bulk distribution coefficient of melting mineral assemblage

        Returns:
        Cl = concentration in the extracted liquid. This is different from the 
        concentration of the instantaneous liquid.
        Cs = concentration in the residual solid

    Note: if Do and P are the same, then you effectively have modal fractional melting
    """

    Cl = (Co / F) * (1 - (1 - F * (P / Do)) ** (1 / P))

    return Cl


# dynamic melting
def non_modal_dynamic_melt(Co, Do, F, P, phi):
    """
    non_modal_dynamic_melt calculates the concentration of a liquid extracted via 
    dynamic melting as described in McKenzie (1985) and Zou (2007) Eq. 3.18. This is 
    applicable for a sitiuation in which melt is in equilibrium when the fraction 
    is below a critical value and then fractional when it is above that value.

    Parameters
    ----------
    Co : array-like
        Concentration of trace element in original solid
    Do : array-like
        Bulk distribution coefficient for element when F = 0
    F : array-like
        fraction of original solid melted (fraction of melt)
    P : array-like
        Bulk distribution coefficient of melting mineral assemblage
    phi : array-like
        critical mass porosity of residue

    Returns
    -------
    Cl : array-like
        Concentration of trace element in the liquid

    """

    X = (F - phi) / (1 - phi)

    Cl = (Co / X) * (
        1
        - (1 - ((X * (P + phi * (1 - P))) / (Do + phi * (1 - P))))
        ** (1 / (phi + (1 - phi) * P))
    )
    return Cl


# crystallization equations
def eq_xtl(
    Cl, D, F,
):
    """
    eq_xtl calculates the composition of a trace element in the remaining liquid after a certain amount of
    crystallization has occured from a source melt when the crystal remeains in equilibrium with the melt
    as described by White (2013) Chapter 7 eq. 7.81. It then calculates the concentration of that trace element
    in a specific solid phase based on partition coefficient input.
        Inputs:
        Cl = concentration of trace element in original liquid
        D = bulk distribution coefficient for trace element of crystallizing assemblage
        F = fraction of melt remaining

        Returns:
        Cl_new = concentration of trace element in the remaining liquid

    """
    Cl_new = Cl / (D + F * (1 - D))
    return Cl_new


# fractional crystallization
def frac_xtl(
    Cl, D, F,
):
    """
    frac_xtl calculates the composition of a trace element in the remaining liquid after a certain amount of
    crystallization has occured from a source melt when the crystal is removed from being in equilibrium with
    the melt as described by White (2013) Chapter 7 eq. 7.82.  It also calculates the 
    concentration of the trace element in the mean cumulate assemblage as described by Rollinson 1993 Eq. 4.20
        Inputs:
        Cl = concentration of trace element in original liquid
        D = bulk distribution coefficient for trace element of crystallizing assemblage
        F = fraction of melt remaining

        Returns:
        Cl_new = concentration of trace element in the remaining liquid
        Cr = concentration in the cumulate
    """
    Cl_new = Cl * (F) ** (D - 1)
    Cr = Cl * ((1 - F ** D) / (1 - F))
    return Cl_new, Cr


# in situ crystallization


def insitu_xtl(Cl, D, F, f, fa):
    """
    insitu_xtl calculates the concentration of the remaining melt as described
    in Langmuir (1989) and Rollinson 1993 Eq. 4.21 whereby crystallization 
    predominantly takes place at the sidewalls of a magma reservoir. Rather than 
    crystals being extracted from the liquid, liquid is extracted from a sidewall
    'mush' in situ. The solidification zone progressively moves through the magma
    chamber until crystallization is complete. In general this amounts in less
    enrichment of incompatible elements and less depletion of compatible elements
    than fractional crystallization

    Parameters
    ----------
    Cl : array-like
        concentration of trace element in original liquid
    D : array-like
        bulk partition coefficient of crystallizing of crystallizing assemblage
    F : array-like
        fraction of melt remaining (between >0 and 1). If 0 is in this array,
        error message will be thrown because python does not do division by 0
    f : array-like
        the fraction of interstitial liquid remaining after crystallization
        within the solidification zone. It is assumed that some of this is 
        trapped in the cumulate (ft) and some is returned to the magma (fa).
        therefore f = ft + fa
    fa : fraction of interstitial liquid that returns to the magma.f = fa would
        be an example where there is no interstital liquid in the crystallization
        front

    Returns
    -------
    Cl_new : array like
        concentration of extracted liquid from crystallization front

    """
    E = 1.0 / (D * (1.0 - f) + f)
    Cl_new = Cl * (F ** ((fa * (E - 1)) / (fa - 1)))
    return Cl_new


def fraclin_xtl(Cl, a, b, F):
    """
    fraclin_xtl calculates the composition of the liquid remaining after it 
    has experienced fractional crystallization where the distribution coefficient
    varies linearly with melt fraction. This was originally described by 
    Greenland 1970.

    Parameters
    ----------
    Cl : array-like
        concentration of the trace element in the original liquid
    a : array-like
        intercept of the relationship describing the linear change in D with melt
        fraction
    b : array-like
        slope of the relationship describing the linear change in D with 
        melt fraction
    F : array-like
        fraction of melt remaining (between 0 and 1). 

    Returns
    -------
    Cl_new : TYPE
        DESCRIPTION.

    """
    Cl_new = Cl * np.exp((a - 1) * np.log(F) + b * (F - 1))
    return Cl_new

def norm_ree(df,source):
    """
    Function to normalize rare earth element data.
    C1: Values taken from McDonough and Sun (1995)
    Primitive Mantle: Values taken from McDonough & Sun (1995)
    Eroded Earth: Values taken from Palme and O'Neill (2003)
    MORB: Values taken from White and Klein (2012)
    ---
    Inputs: 
    Dataframe with ree La through Lu. Values must be in ppm
    Source: Reservoir you wish to normalize your data to. These are listed above.
    ---
    Outputs: The input dataframe with normalized values included with a x_N as the column header, where x is the element.
    
    """
    import numpy as np
    
    # make arrays of compositions
    C1 = np.array([[0.237, 0.613, 0.0928, 0.457, 0.148, 0.0563, 0.199, 0.0361, 0.246, 0.0546, 0.160, 0.0247, 0.161, 0.0246]])
    prim_mntl = np.array([[0.648, 0.68, 0.254, 1.25, 0.406, 0.154, 0.544, 0.099, 0.674, 0.149, 0.438, 0.068, 0.441, 0.068]])
    eroded_Earth = np.array([[0.555, 1.53, 0.235, 1.16, 0.389, 0.147, 0.523, 0.097, 0.666, 0.149, 0.440, 0.068]])
    morb = np.array([[4.87, 5.81, 0.94, 4.9, 1.70, 0.62, 2.25, 0.43, 2.84, 0.63, 1.85, 0.28, 1.85, 0.28]])    
    
    to_calc = df.loc[0::,'La':'Lu']
    
    if source == "C1":
        ree_norm = to_calc / C1
    elif source == "Primitive Mantle":
        ree_norm = to_calc / prim_mntl
    elif source == "Eroded Earth":
        ree_norm = to_calc / eroded_Earth
    elif source == "MORB":
        ree_norm = to_calc / morb
    else:
        print('Check Source Inputs')     
    
    df[['la_N','ce_N','pr_N','nd_N','sm_N','eu_N','gd_N','tb_N','dy_N','ho_N','er_N','tm_N','yb_N','lu_N']] = ree_norm
    
    return df

def norm_spider(df,source):
    """
    Function to normalize rare earth element data.
    C1: Values taken from McDonough and Sun (1995)
    Primitive Mantle: Values taken from McDonough & Sun (1995)
    Eroded Earth: Values taken from Palme and O'Neill (2003)
    MORB: Values taken from White and Klein (2012)
    ---
    Inputs: 
    Dataframe with trace elements. Values must be in ppm. Elements included are:
    Cs, Rb, Ba, Th, U, Nb, Ta, K, La, Ce, Pb, Pr, Sr, Nd, Sm, Zr, Hf, Eu, Gd, Tb, Dy, Y, Ho, Er, Tm, Yb, Lu
    in that order.
    Source: Reservoir you wish to normalize your data to. These are listed above.
    ---
    Outputs: The input dataframe with normalized values included with a x_N as the column header, where x is the element.
    
    """
    import numpy as np
    
    # make arrays of compositions
    C1 = np.array([[0.190, 2.30, 2.410, 0.029, 0.0074, 0.240, 0.0136, 550, 0.237, 0.613, 2.470, 0.0928, 7.25, 0.457, 0.148, 3.82, 0.103, 0.0563, 0.199,
                   0.0361, 0.246, 1.57, 0.0546, 0.160, 0.0247, 0.161, 0.0246]])
    prim_mntl = np.array([[0.021, 0.60, 6.60, 0.080, 0.02, 0.66, 0.037, 240, 0.648, 1.68, 0.150, 0.254, 19.90, 1.25, 0.406, 10.50, 0.283, 0.154, 0.544, 0.099,
                          0.674, 4.30, 0.149, 0.438, 0.068, 0.441, 0.068]])
    eroded_Earth = np.array([[0.015, 0.47, 5.03, 0.063, 0.0164, 0.45, 0.031, 226, 0.555, 1.53, 0.120, 0.235, 17.48, 1.16, 0.389, 9.64, 0.269, 0.147, 0.523,
                             0.097, 0.666, 4.12, 0.149, 0.440, 0.068, 0.440, 0.068]])
    morb = np.array([[0.05, 4.05, 43.4, 0.491, 0.157, 6.44, 0.417, 1237, 4.87, 13.1, 0.657, 2.08, 138, 10.4, 3.37, 103, 2.62, 1.20, 4.42, 0.81, 5.28, 32.4,
                     1.14, 3.30, 0.49, 3.17, 0.48]])
    
    to_calc = df.loc[0::,'Cs':'Lu']
    
    if source == "C1":
        ree_norm = to_calc / C1
    elif source == "Primitive Mantle":
        ree_norm = to_calc / prim_mntl
    elif source == "Eroded Earth":
        ree_norm = to_calc / eroded_Earth
    elif source == "MORB":
        ree_norm = to_calc / morb
    else:
        print('Check Source Inputs')     
    
    df[['Cs_N','Rb_N','Ba_N','Th_N','U_N','Nb_N','Ta_N','K_N','La_N','Ce_N','Pb_N','Pr_N','Sr_N','Nd_N', 'Sm_N', 'Zr_N', 'Hf_N', 'Eu_N', 'Gd_N', 'Tb_N',
       'Dy_N', 'Y_N', 'Ho_N', 'Er_N', 'Tm_N', 'Yb_N', 'Lu_N']] = ree_norm
    
    return df


#%% General mineral recalculation.
def mineral_formula_calc(df, n_oxygens, mineral, normalized,index):

    """
    mineral_formula_calc is a function that calculates the stoichiometry for a mineral based on a set of major
    element oxide analyses as described by Deer et al., 1966 Appendix 1
    
    Inputs:
    df : pandas dataframe object of major element analyses. Column headers must have the the element somewhere in the name
    
    ** if a column containing 'Total' in the name exists, it will be removed so that only the individual analyses are 
    present
    ** your dataframe should have a column that pertains to sample, analysis number, etc. This will be set as the index
    of the dataframe so that chemical formulas can be accessed easily upon calculation
    
    EXAMPLE OF INPUT DATAFRAME:
    |sample|SiO2|TiO2|Al2O3|Cr2O3|FeO|BaO|SrO|MnO|CaO|Na2O|K2O|NiO|Total| <---- currently supported elements
    
    
    n_oxygens : number of ideal oxygens in the chemical formula (e.g., for feldspars this would be 8)
    
    mineral : 'feldspar','olivine','pyroxene'
    if 'pyroxene' is chosen, the function will calculate the proportions of Fe2+ and Fe3+ based off stoichiometry and charge
    balance as described by Droop 1987. If 'feldspar', all Fe is assumed to be Fe3+. If 'olivine', all Fe is assumed to be 2+
    
    normalized: boolean 
    if True, will normalize your geochemical analyses. If false, mineral formulas will be calculated using 
    raw geochemical data
    
    index: string
    column denoting which column to be used as the index for the dataframe. Suggested that this is a column that 
    denotes sample name or spot name or something similar
    
    
    Returns:
    norm_cations: pandas dataframe object that contains the calculated number of cations in the chemical formula
    normalized to the amount of ideal oxygens specified by 'n_oxygens'. 
    
    
    
    """

    data = df.copy()
    data.set_index(index,inplace = True)
    data.fillna(0, inplace=True)
    # if index is not None:

    #     data.set_index(index,inplace = True)
    # else:
    #     data.index = np

    # Removes the 'total column' from the list
    columns = list(data.columns)
    elements = []
    for column in columns:
        if "Total" in column:
            columns.remove(column)

    # can make this a delimeter variable for the user to choose from
    # dropping anything after the underscore
    for column in columns:

        if "Si" in column:
            elements.append(column.split("_")[0])
        if "Ti" in column:
            elements.append(column.split("_")[0])
        if "Al" in column:
            elements.append(column.split("_")[0])
        if "Cr" in column:
            elements.append(column.split("_")[0])
        if "Fe" in column:
            elements.append(column.split("_")[0])
        if "Ba" in column:
            elements.append(column.split("_")[0])
        if "Sr" in column:
            elements.append(column.split("_")[0])
        if "Mn" in column:
            elements.append(column.split("_")[0])
        if "Mg" in column:
            elements.append(column.split("_")[0])
        if "Na" in column:
            elements.append(column.split("_")[0])
        if "K" in column:
            elements.append(column.split("_")[0])
        if "Ca" in column:
            elements.append(column.split("_")[0])
        if "Ni" in column:
            elements.append(column.split("_")[0])
        if "Cl" in column:
            elements.append(column.split("_")[0])
        if "P2O5" in column:
            elements.append(column.split("_")[0])

    # create new dataframe that is just the analyses without the total
    oxides = data.loc[:, columns]
    oxides.columns = elements

    if normalized == True:

        # normalize the wt%
        oxides_normalized = 100 * (oxides.div(oxides.sum(axis="columns"), axis="rows"))
    elif normalized == False:
        oxides_normalized = oxides.copy()

    # create an array filled with zeros such that it is the same shape of our input
    # data
    mol_cations = np.zeros(oxides_normalized.shape)

    # these loops are saying that: for each element in my list of elements (e.g., columns)
    # check to see if the given string (e.g., Si) is in it. If it is, then populate that column
    # of the array with the appropriate math

    # Here we call on the mendeleev package module 'element' to get the mass from a given element
    # e.g.(el(element).mass)
    for i, element in zip(range(len(elements)), elements):
        if "Si" in element:
            mol_cations[:, i] = oxides_normalized[element] / (28.09 + (16 * 2))
        elif "Ti" in element:
            mol_cations[:, i] = oxides_normalized[element] / (47.87 + (16 * 2))
        elif "Al" in element:
            mol_cations[:, i] = (2 * oxides_normalized[element]) / (
                (26.98 * 2) + (16 * 3)
            )
        elif "Cr" in element:
            mol_cations[:, i] = (2 * oxides_normalized[element]) / ((52 * 2) + (16 * 3))
        elif "Fe" in element:
            mol_cations[:, i] = oxides_normalized[element] / (55.85 + 16)
        elif "Ba" in element:
            mol_cations[:, i] = oxides_normalized[element] / (137.33 + 16)
        elif "Sr" in element:
            mol_cations[:, i] = oxides_normalized[element] / (87.62 + 16)
        elif "Mn" in element:
            mol_cations[:, i] = oxides_normalized[element] / (54.94 + 16)
        elif "Mg" in element:
            mol_cations[:, i] = oxides_normalized[element] / (24.31 + 16)
        elif "Ca" in element:
            mol_cations[:, i] = oxides_normalized[element] / (40.08 + 16)
        elif "Na" in element:
            mol_cations[:, i] = (2 * oxides_normalized[element]) / ((23 * 2) + 16)
        elif "K" in element:
            mol_cations[:, i] = (2 * oxides_normalized[element]) / ((39.1 * 2) + 16)
        elif "Ni" in element:
            mol_cations[:, i] = oxides_normalized[element] / (58.69 + 16)

    mol_cations = pd.DataFrame(mol_cations, columns=elements)

    # Calculating the number of oxygens per cation in the formula
    mol_oxygens = np.zeros(mol_cations.shape)

    for i, element in zip(range(len(elements)), elements):
        if "Si" in element:
            mol_oxygens[:, i] = mol_cations[element] * 2
        elif "Ti" in element:
            mol_oxygens[:, i] = mol_cations[element] * 2
        elif "Al" in element:
            mol_oxygens[:, i] = mol_cations[element] * (3 / 2)
        elif "Cr" in element:
            mol_oxygens[:, i] = mol_cations[element] * (3 / 2)
        elif "Fe" in element:
            mol_oxygens[:, i] = mol_cations[element] * 1
        elif "Ba" in element:
            mol_oxygens[:, i] = mol_cations[element] * 1
        elif "Sr" in element:
            mol_oxygens[:, i] = mol_cations[element] * 1
        elif "Mn" in element:
            mol_oxygens[:, i] = mol_cations[element] * 1
        elif "Mg" in element:
            mol_oxygens[:, i] = mol_cations[element] * 1
        elif "Ca" in element:
            mol_oxygens[:, i] = mol_cations[element] * 1
        elif "Na" in element:
            mol_oxygens[:, i] = mol_cations[element] * (1 / 2)
        elif "K" in element:
            mol_oxygens[:, i] = mol_cations[element] * (1 / 2)
        elif "Ni" in element:
            mol_oxygens[:, i] = mol_cations[element] * 1

    mol_oxygens = pd.DataFrame(mol_oxygens, columns=elements)

    # number of oxygens per cation, normalized to the ideal number of oxygens specified above
    norm_oxygens = (mol_oxygens * n_oxygens).div(
        mol_oxygens.sum(axis="columns"), axis="rows"
    )

    # calculate the mole cations of each oxide normalized to the number of ideal oxygens
    norm_cations = np.zeros(norm_oxygens.shape)

    for i, element in zip(range(len(elements)), elements):
        if "Si" in element:
            norm_cations[:, i] = norm_oxygens[element] / 2
        elif "Ti" in element:
            norm_cations[:, i] = norm_oxygens[element] / 2
        elif "Al" in element:
            norm_cations[:, i] = norm_oxygens[element] / (3 / 2)
        elif "Cr" in element:
            norm_cations[:, i] = norm_oxygens[element] / (3 / 2)
        elif "Fe" in element:
            norm_cations[:, i] = norm_oxygens[element]
        elif "Ba" in element:
            norm_cations[:, i] = norm_oxygens[element]
        elif "Sr" in element:
            norm_cations[:, i] = norm_oxygens[element]
        elif "Mn" in element:
            norm_cations[:, i] = norm_oxygens[element]
        elif "Mg" in element:
            norm_cations[:, i] = norm_oxygens[element]
        elif "Ca" in element:
            norm_cations[:, i] = norm_oxygens[element]
        elif "Na" in element:
            norm_cations[:, i] = norm_oxygens[element] / (1 / 2)
        elif "K" in element:
            norm_cations[:, i] = norm_oxygens[element] / (1 / 2)
        elif "Ni" in element:
            norm_cations[:, i] = norm_oxygens[element]

    cations = []
    # Get the cations by taking the first two characters
    [cations.append(element[:2]) for element in elements]

    # since some elements are only one letter (e.g., K) this
    # strips the number from it

    r = re.compile("([a-zA-Z]+)([0-9]+)")
    for i in range(len(cations)):

        m = r.match(cations[i])
        if m != None:
            cations[i] = m.group(1)

    norm_cations = pd.DataFrame(norm_cations,columns = cations)
    norm_cations['Total_cations'] = norm_cations.sum(axis = 'columns')
    norm_cations[data.index.name] = data.index.tolist()

    if mineral == "pyroxene":
        # ideal cations
        T = 4

        # calculated cations based on oxide measurements
        # S = norm_cations['Total_cations']

        # step 2 and 3 from Droop 1987
        norm_cations.loc[norm_cations["Total_cations"] > T, "Fe_3"] = (
            2 * n_oxygens * (1 - (T / norm_cations["Total_cations"]))
        )
        norm_cations.loc[norm_cations["Total_cations"] <= T, "Fe_3"] = 0

        # step 4 from Droop 1987
        norm_cations.set_index(data.index.name, inplace=True)

        ts = T / norm_cations["Total_cations"].to_numpy()
        norm_cations = norm_cations * ts[:, np.newaxis]

        norm_cations["Fe_2"] = norm_cations["Fe"] - norm_cations["Fe_3"]

    else:

        norm_cations.set_index(data.index.name, inplace=True)

    return norm_cations


def hb_plag_amph_temp(plag_cations, amp_sites_fe, P, thermometer):

    """
    hb_plag_amph_temp uses the Holland and Blundy (1994) equations to calculate 
    temperatures of formation for plagioclase - amphibole pairs. 
    
    Thermometer A: for use in assemblages where plagiocalse and amphibole are 
    co crystallizing with quartz
    
    Thermometer B: for use in assemblages where plagioclase and amphibole are 
    crystallizing without quartz
    
    Inputs: 
    
    plag_cations : pandas DataFrame
    a dataframe consisting of plagioclase cation values. 
    
    amp_sites_fe : pandas DataFrame
    a dataframe consiting of ideal site assignments that includes ferric and ferrous
    iron. This is the output from the "get_amp_sites_ferric_ferrous" function and does
    not need any tweaking
    
    P : scalar
    Pressure of formation in kbar
    
    thermometer : string
    Which thermometer you would like to use: Either "A" or "B"
    
    Returns:
    
    T_df: pandas DataFrame
    dataframe of temperature calculation results for each grain in the input dataframes.
    Where there are multiple analyses per phase per grain, every possible temperature 
    will be calculated (e.g., 4 amphibole analyes and 3 plag analyses per grain/sample
    would yield 12 temperatures)
    
    
    """

    # thermodynamic parameters
    R = 0.0083144  # kJ/K
    Ab = (
        plag_cations["Na"]
        / (plag_cations["Na"] + plag_cations["Ca"] + plag_cations["K"])
    ).to_numpy()
    An = (
        plag_cations["Ca"]
        / (plag_cations["Na"] + plag_cations["Ca"] + plag_cations["K"])
    ).to_numpy()

    plag_cations["An"] = An
    plag_cations["Ab"] = Ab
    # Calculating Yab for Thermometer A
    Y = np.empty(An.shape)

    # Yab-an parameters for each thermometer
    for i in range(len(An)):
        # Calculating Yab for Thermometer A

        if thermometer == "A":

            if Ab[i] > 0.5:
                Y[i] = 0
            else:
                Y[i] = 12.01 * (1 - Ab[i]) ** 2 - 3

        elif thermometer == "B":

            # Calculating Yab-an for Thermometer B
            if Ab[i] > 0.5:
                Y[i] = 3
            else:
                Y[i] = 12.0 * (2 * Ab[i] - 1) + 3
        else:
            raise Exception(
                'This alphabet is only two letters long. Please choose "A" or "B" for your thermometer'
            )
    plag_cations["Y"] = Y
    # cummingtonite substitution

    cm = (
        amp_sites_fe["Si_T"]
        + (amp_sites_fe["Al_T"] + amp_sites_fe["Al_D"])
        + amp_sites_fe["Ti_D"]
        + amp_sites_fe["Fe3_D"]
        + (amp_sites_fe["Fe2_C"] + amp_sites_fe["Fe2_B"])
        + (amp_sites_fe["Mg_D"] + amp_sites_fe["Mg_C"])
        + amp_sites_fe["Mn_C"]
        + amp_sites_fe["Mn_B"]
        - 13
    )

    # site terms for the thermometer
    Si_T1 = (amp_sites_fe["Si_T"] - 4) / 4

    Al_T1 = (8 - amp_sites_fe["Si_T"]) / 4

    Al_M2 = (amp_sites_fe["Al_T"] + amp_sites_fe["Al_D"] + amp_sites_fe["Si_T"] - 8) / 2

    K_A = amp_sites_fe["K_A"]

    box_A = (
        3
        - amp_sites_fe["Ca_B"]
        - (amp_sites_fe["Na_B"] + amp_sites_fe["Na_A"])
        - amp_sites_fe["K_A"]
        - cm
    )

    Na_A = amp_sites_fe["Ca_B"] + amp_sites_fe["Na_B"] + amp_sites_fe["Na_A"] + cm - 2

    Na_M4 = (2 - amp_sites_fe["Ca_B"] - cm) / 2

    Ca_M4 = amp_sites_fe["Ca_B"] / 2

    hbl_plag_params = pd.DataFrame(
        {
            "Si_T1": Si_T1,
            "Al_T1": Al_T1,
            "Al_M2": Al_M2,
            "K_A": K_A,
            "box_A": box_A,
            "Na_A": Na_A,
            "Na_M4": Na_M4,
            "Ca_M4": Ca_M4,
        }
    )
    # put the index back in for the sample labels
    hbl_plag_params.index = amp_sites_fe.index

    # checks for the same unique index names in your plag and amphibole dataframes
    sameset = set(hbl_plag_params.index.unique().to_list())

    samegrains = list(sameset.intersection(plag_cations.index.unique().to_list()))

    # empty list to fill with individual temperature dataframes
    T_df_list = []

    for grain in samegrains:

        # this extracts each individual grain from the respective dataframes
        # for plag and amphibole data
        amp = hbl_plag_params.loc[grain, :]
        plag = plag_cations.loc[grain, :]

        # This conditional checks how many plag analyses there are for a given grain.
        # if there is more than one it will follow the first option and use array broadcasting
        # to calculate every possible temperature for a given amphibole - plagioclase pair.
        # e.g. if you have two plag analyses and 4 amphibole analyses you will get 8 total temperatures
        if len(amp.shape) == 2:

            if len(plag.shape) == 2:
                if thermometer == "A":

                    # numerator for thermometer A
                    top = (
                        -76.95
                        + (0.79 * P)
                        + plag["Y"].to_numpy()[:, np.newaxis]
                        + 39.4 * amp["Na_A"].to_numpy()
                        + 22.4 * amp["K_A"].to_numpy()
                        + (41.5 - 2.89 * P) * amp["Al_M2"].to_numpy()
                    )
                    # denominator for thermometer A
                    bottom = -0.0650 - R * np.log(
                        (
                            27
                            * amp["box_A"].to_numpy()
                            * amp["Si_T1"].to_numpy()
                            * plag["Ab"].to_numpy()[:, np.newaxis]
                        )
                        / (256 * amp["Na_A"].to_numpy() * amp["Al_T1"].to_numpy())
                    )

                    # final thermometer A
                    T = (top / bottom) - 273.15
                elif thermometer == "B":

                    # thermometer B whole thing
                    T = (
                        (
                            78.44
                            + plag["Y"].to_numpy()[:, np.newaxis]
                            - 33.6 * amp["Na_M4"].to_numpy()
                            - (66.8 - 2.92 * P) * amp["Al_M2"].to_numpy()
                            + 78.5 * amp["Al_T1"].to_numpy()
                            + 9.4 * amp["Na_A"].to_numpy()
                        )
                        / (
                            0.0721
                            - R
                            * np.log(
                                (
                                    27
                                    * amp["Na_M4"].to_numpy()
                                    * amp["Si_T1"].to_numpy()
                                    * plag["An"].to_numpy()[:, np.newaxis]
                                )
                                / (
                                    64
                                    * amp["Ca_M4"].to_numpy()
                                    * amp["Al_T1"].to_numpy()
                                    * plag["Ab"].to_numpy()[:, np.newaxis]
                                )
                            )
                        )
                    ) - 273.15

                # making the temperatures for a given grain dataframe for ease of use later on
                T_df_list.append(
                    pd.DataFrame({"grain": grain, "T": np.concatenate(T, axis=None),})
                )

            # This is triggered if there is only one plag analysis per amphibole. In this case
            # we don't need array broadcasting because the plag An/Ab variables are scalars. All the
            # equations are the same as above

            else:
                if thermometer == "A":

                    top = (
                        -76.95
                        + (0.79 * P)
                        + plag["Y"]
                        + 39.4 * amp["Na_A"].to_numpy()
                        + 22.4 * amp["K_A"].to_numpy()
                        + (41.5 - 2.89 * P) * amp["Al_M2"].to_numpy()
                    )
                    bottom = -0.0650 - R * np.log(
                        (
                            27
                            * amp["box_A"].to_numpy()
                            * amp["Si_T1"].to_numpy()
                            * plag["Ab"]
                        )
                        / (256 * amp["Na_A"].to_numpy() * amp["Al_T1"].to_numpy())
                    )

                    T = (top / bottom) - 273.15
                elif thermometer == "B":

                    T = (
                        (
                            78.44
                            + plag["Y"]
                            - 33.6 * amp["Na_M4"].to_numpy()
                            - (66.8 - 2.92 * P) * amp["Al_M2"].to_numpy()
                            + 78.5 * amp["Al_T1"].to_numpy()
                            + 9.4 * amp["Na_A"].to_numpy()
                        )
                        / (
                            0.0721
                            - R
                            * np.log(
                                (
                                    27
                                    * amp["Na_M4"].to_numpy()
                                    * amp["Si_T1"].to_numpy()
                                    * plag["An"]
                                )
                                / (
                                    64
                                    * amp["Ca_M4"].to_numpy()
                                    * amp["Al_T1"].to_numpy()
                                    * plag["Ab"]
                                )
                            )
                        )
                    ) - 273.15

                T_df_list.append(
                    pd.DataFrame({"grain": grain, "T": np.concatenate(T, axis=None),})
                )
        # This is triggered if there is only one amphibole analysis per plag. In this case
        # we don't need array broadcasting or the .to_numpy() function because the amphibole
        # values are already scalars
        else:

            if len(plag.shape) == 2:

                if thermometer == "A":

                    # numerator for thermometer A
                    top = (
                        -76.95
                        + (0.79 * P)
                        + plag["Y"][:, np.newaxis]
                        + 39.4 * amp["Na_A"]
                        + 22.4 * amp["K_A"]
                        + (41.5 - 2.89 * P) * amp["Al_M2"]
                    )
                    # denominator for thermometer A
                    bottom = -0.0650 - R * np.log(
                        (27 * amp["box_A"] * amp["Si_T1"] * plag["Ab"][:, np.newaxis])
                        / (256 * amp["Na_A"].to_numpy() * amp["Al_T1"])
                    )

                    # final thermometer A
                    T = (top / bottom) - 273.15

                elif thermometer == "B":
                    # thermometer B whole thing
                    T = (
                        (
                            78.44
                            + plag["Y"][:, np.newaxis]
                            - 33.6 * amp["Na_M4"]
                            - (66.8 - 2.92 * P) * amp["Al_M2"]
                            + 78.5 * amp["Al_T1"]
                            + 9.4 * amp["Na_A"]
                        )
                        / (
                            0.0721
                            - R
                            * np.log(
                                (
                                    27
                                    * amp["Na_M4"]
                                    * amp["Si_T1"]
                                    * plag["An"][:, np.newaxis]
                                )
                                / (
                                    64
                                    * amp["Ca_M4"]
                                    * amp["Al_T1"]
                                    * plag["Ab"][:, np.newaxis]
                                )
                            )
                        )
                    ) - 273.15

                # making the temperatures for a given grain dataframe for ease of use later on
                T_df_list.append(
                    pd.DataFrame({"grain": grain, "T": np.concatenate(T, axis=None),})
                )

            # This is triggered if there is only one plag analysis per amphibole. In this case
            # we don't need array broadcasting because the plag An/Ab variables are scalars. All the
            # equations are the same as above

            else:
                if thermometer == "A":

                    top = (
                        -76.95
                        + (0.79 * P)
                        + plag["Y"]
                        + 39.4 * amp["Na_A"]
                        + 22.4 * amp["K_A"]
                        + (41.5 - 2.89 * P) * amp["Al_M2"]
                    )
                    bottom = -0.0650 - R * np.log(
                        (27 * amp["box_A"] * amp["Si_T1"] * plag["Ab"])
                        / (256 * amp["Na_A"] * amp["Al_T1"])
                    )

                    T = (top / bottom) - 273.15
                elif thermometer == "B":

                    T = (
                        (
                            78.44
                            + plag["Y"]
                            - 33.6 * amp["Na_M4"]
                            - (66.8 - 2.92 * P) * amp["Al_M2"]
                            + 78.5 * amp["Al_T1"]
                            + 9.4 * amp["Na_A"]
                        )
                        / (
                            0.0721
                            - R
                            * np.log(
                                (27 * amp["Na_M4"] * amp["Si_T1"] * plag["An"])
                                / (64 * amp["Ca_M4"] * amp["Al_T1"] * plag["Ab"])
                            )
                        )
                    ) - 273.15

                T_df_list.append(pd.DataFrame({"grain": grain, "T_calc": [T],},))

    # overall temperature dataframe for every grain
    T_df = pd.concat(T_df_list)
    return T_df


#%% Cation fraction calculation for barometry based on Putirka 2008
