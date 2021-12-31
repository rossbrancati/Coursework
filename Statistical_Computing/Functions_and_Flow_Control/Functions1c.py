#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ross Brancati
"""

def compute_rate(annual_salary):
    """Compute the savings rate needed"""
    
    # TODO: Implement me
    #assign asssumptions to variables
    semi_annual_raise = 0.07
    r = 0.04 #annual investment return
    portion_down_payment = 0.25
    total_cost = 1000000
    
    #calculate the total amount needed for down payment
    down_payment = total_cost * portion_down_payment
    
    #initialize high and low bounds for bisection
    low = 0
    high = 10000
    
    #start with some guess of what rate will achieve enough savings in 36 months
    guess = (high+low)//2
    
    def calculate_savings(current_savings, annual_salary, guess):
        
        #reset current savings to 0 each time you call this function
        current_savings = 0
        #loop over 36 months, or 3 years
        for month in range(36):
            #check if month index is not = 0 and divisible by 6 will only edit the annual income every 6 months, and ignore the first month because 6/0 = 0
            if month != 0 and month % 6 == 0:
                annual_salary = annual_salary + annual_salary*semi_annual_raise
            #convert rate to decimal from guess
            rate = guess/10000
            #calculate the monthly salary
            monthly_savings = (annual_salary/12) * rate
            #calcualte the monthly return on investment
            return_on_investment = current_savings*(r/12)
            #calculate the current_savings up to the current month
            current_savings = current_savings + monthly_savings + return_on_investment
            
        return current_savings
        
    
    #start with 0 current_savings
    current_savings = 0
    #initialize the biseciton counter
    bisection_count = 0
    #check if savings are within $100 of down payment
    while abs(current_savings-down_payment) >= 100:
        #count the number of bisections to achieve lowest rate
        bisection_count += 1
        #check if the annual salary is too low by checking if you saved 100% of the annual salary would be less than the down payment
        if (annual_salary * 3) < down_payment:
            rate = None
            bisection_count = None
            break
        #calculate the current_savings from calculate_savings function
        current_savings = calculate_savings(current_savings, annual_salary, guess)
        #if the current savings are less than the down payment, increase the minimal savings rate
        if current_savings < down_payment:
            low = guess
        #if the current savings are greater than the down payment, increase the minimal savings rate
        elif current_savings > down_payment:
            high = guess
        #update the initial guess based on results of this 36 month calculation
        guess = (high + low)//2
        #convert rate to decimal from guess
        rate = round((guess/10000), 4)
        
    return rate, bisection_count
    

    
if __name__ == "__main__":
    annual_salary = int(input("Enter your starting annual salary: "))
    
    rate, bisection_count = compute_rate(annual_salary)
    if rate is None:
        print("It is not possible to pay the down payment in three years.")
    else:
        print("Minimal savings rate: %f" % rate)
        print("Steps in bisection search: %d" % bisection_count)
    
