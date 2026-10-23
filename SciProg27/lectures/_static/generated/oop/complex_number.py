import math


class ComplexNumber:

    def __init__(self, real, imaginary):
        raise Exception("Implement me!")

    def __str__(self):
        raise Exception("Implement me!")

    def phase(self):
        """ Returns a float which is the phase (that is, the vector angle) of the complex number 
        
            This method is something we introduce by ourselves, according to the definition:
            https://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
        """
        raise Exception("Implement me!")    

    def magnitude(self):
        """ Returns a float which is the magnitude (that is, the absolute value) of the complex number 
        
            This method is something we introduce by ourselves, according to the definition:
            https://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
        """
        raise Exception("Implement me!")
    
    
    def log(self, base):
        """ Returns another ComplexNumber which is the logarithm of this complex number 
            
            This method is something we introduce by ourselves, according to the definition:
            (accomodated for generic base b)
            https://en.wikipedia.org/wiki/Complex_number#Natural_logarithm
        """      
        raise Exception("Implement me!")
    
    
    def isclose(self, z, delta):
        """ Returns True if the complex number is within a delta distance from complex number z.                                
        """
        raise Exception("Implement me!")
    
    
    def __eq__(self, other):         
        raise Exception("Implement me!")
        
    
        
    def __add__(self, other):         
        raise Exception("Implement me!")
  

    def __radd__(self, other):         
        
        raise Exception("Implement me!")
     
    
    def __mul__(self, other): 
        
        raise Exception("Implement me!")
        
    def __rmul__(self, other):        
        raise Exception("Implement me!")
        
 