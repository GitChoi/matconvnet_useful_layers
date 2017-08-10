classdef my_extractch < dagnn.ElementWise
    % Edited by GitChoi
    % A layer to extract certain channels from the previous layer
    
    properties
        ch
    end
    
    properties (Transient)
        
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = inputs{1}(:,:,obj.ch,:);
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = 0*inputs{1};
            derInputs{1}(:,:,obj.ch,:) = derOutputs{1};
            
            derParams = {} ;
        end
        
        function obj = my_extractch(varargin)
            obj.load(varargin) ;
        end
    end
end
