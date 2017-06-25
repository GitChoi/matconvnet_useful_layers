classdef my_fastSoftmax < dagnn.ElementWise
    % Edited by GitChoi
    % Faster version of dagnn SoftMax layer 
    % Faster backward prop. speed (almost x2)
    
    properties
        
    end
    
    properties (Transient)
        Y       % temporary save for faster backward prop. speed
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            
            outputs{1} = exp(bsxfun(@minus, inputs{1}, max(inputs{1},[],3))) ;
            L = sum(outputs{1},3) ;
            outputs{1} = bsxfun(@rdivide, outputs{1}, L) ;
            
            obj.Y = outputs{1};
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            derInputs{1} = obj.Y .* bsxfun(@minus, derOutputs{1}, sum(derOutputs{1} .* obj.Y, 3)) ;
            
            derParams = {} ;
        end
        
        function obj = my_fastSoftmax(varargin)
            obj.load(varargin) ;
        end
    end
end
