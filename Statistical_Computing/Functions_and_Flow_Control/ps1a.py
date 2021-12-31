#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ross Brancati 
"""

def compute_months(annual_salary, portion_saved, total_cost):
    """Compute the number of months needed to save."""

    # TODO: Implement me
    #calculate the cost of the down payment on the house
    portion_down_payment = 0.25 * total_cost
    #starting with a current savings of 0
    current_savings = 0
    #initialize month counter
    month = 0
    #4% annual return constant
    r = 0.04 
    
    #loop that holds true when the current savings are less than the down payment
    while current_savings <= portion_down_payment:
        #check if house is free and return 0 months save up period
        if portion_down_payment == 0:
            month = 0
            break
        #calculate money saved each month from portion of salary
        monthly_savings = (annual_salary/12) * portion_saved
        #calcualte the monthly return on investment
        return_on_investment = (current_savings*r)/12
        #calculate the current_savings up to the current month
        current_savings = current_savings + monthly_savings + return_on_investment
        #count the number of months in the loop, adding one month for each month of saving
        month = month + 1
    
    return month
    
    
if __name__ == "__main__":
    """
    annual_salary is your annual salaray in dollars
    portion_saved is the amount of your salaray to dedicate to savings per month 
                  to the down payment in decimal form (ie 0.1 for 10%)
    total_cost is the total cost of the prospective house in dollars
    """
    annual_salary = int(input("Enter your starting annual salary: "))
    # TODO: Implement me
    
    #assign user inputs to portion_saved and total_cost as floats
    portion_saved = float(input('Enter the percent of your salary to save, as a decimal: '))
    total_cost = float(input('Enter the total cost of your house: '))
    
    num_months = compute_months(annual_salary, portion_saved, total_cost)
    print("Number of months: %d" % num_months)
    
