classdef my_SeLU < dagnn.ElementWise
    % Edited by GitChoi
    % SeLU from Self-Normalizing Neural Networks
    % https://arxiv.org/abs/1706.02515
    
    properties
        l = 1       % alpha
        a = 0.1     % lambda
    end
    
    properties (Transient)
        expx
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.expx = exp(inputs{1});
            
            outputs{1} = obj.l*(max(inputs{1},0) + (obj.a*obj.expx - obj.a).*(inputs{1}<0));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = obj.l*((inputs{1}>=0) + (obj.a*obj.expx).*(inputs{1}<0)).*derOutputs{1};
            
            derParams = {} ;
        end
        
        function obj = my_SeLU(varargin)
            obj.load(varargin) ;
        end
    end
end
