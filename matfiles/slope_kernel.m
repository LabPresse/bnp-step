    function y = slope_kernel(t)
        slope = 0.25;
        d     = 20  ; 
        y = t * slope;
        y(t > 1/slope) = 1;
        y((0>t)|(t > d)) = 0;


    end