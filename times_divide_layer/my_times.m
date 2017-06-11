classdef my_times < dagnn.ElementWise
    %SUM DagNN sum layer
    %   The SUM layer takes the sum of all its inputs and store the result
    %   as its only output.
    
    % Edited by GitChoi
    % Element-wise times layer
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = inputs{1}.*inputs{2};
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = derOutputs{1}.*inputs{2};
            derInputs{2} = inputs{1}.*derOutputs{1};
            
            derParams = {} ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = inputSizes{1} ;
            for k = 2:numel(inputSizes)
                if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
                    if ~isequal(inputSizes{k}, outputSizes{1})
                        warning('Sum layer: the dimensions of the input variables is not the same.') ;
                    end
                end
            end
        end
        
        function obj = my_times(varargin)
            obj.load(varargin) ;
        end
    end
end
