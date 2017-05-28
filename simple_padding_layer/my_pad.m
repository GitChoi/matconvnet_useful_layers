classdef my_pad < dagnn.ElementWise
    % Edited by GitChoi
    % DagNN layer
    % Simple padding layer
    
    properties
        pp = 0
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = padarray(inputs{1},[obj.pp,obj.pp,0,0],'replicate');
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = derOutputs{1}(1+obj.pp:end-obj.pp,1+obj.pp:end-obj.pp,:,:);
            
            derParams = {} ;
        end
        
        function obj = my_pad(varargin)
            obj.load(varargin) ;
        end
    end
end
