classdef my_bsxTimes < dagnn.ElementWise
    %SUM DagNN sum layer
    %   The SUM layer takes the sum of all its inputs and store the result
    %   as its only output.
    
    % Edited by GitChoi
    % A layer functioning as MATLAB bsxfun @times
    
    properties (Transient)
        numInputs
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs) ;
            outputs{1} = bsxfun(@times,inputs{1},inputs{2});
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = bsxfun(@times,derOutputs{1},inputs{2});
            derInputs{2} = bsxfun(@times,inputs{1},derOutputs{1});
            if size(inputs{1},3)==1
                derInputs{1} = sum(derInputs{1},3);
            end
            if size(inputs{2},3)==1
                derInputs{2} = sum(derInputs{2},3);
            end
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
        
        function rfs = getReceptiveFields(obj)
            numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
            rfs.size = [1 1] ;
            rfs.stride = [1 1] ;
            rfs.offset = [1 1] ;
            rfs = repmat(rfs, numInputs, 1) ;
        end
        
        function obj = my_bsxTimes(varargin)
            obj.load(varargin) ;
        end
    end
end
