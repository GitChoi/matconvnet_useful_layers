classdef my_ESPC < dagnn.ElementWise
    %SUM DagNN sum layer
    %   The SUM layer takes the sum of all its inputs and store the result
    %   as its only output.
    
    % Edited by GitChoi
    % Sub-pixel convolutional layer from
    % Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    % CVPR 2016
    
    properties (Transient)
        numInputs
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs) ;
            sc = sqrt(size(inputs{1},3));
            outputs{1} = zeros([size(inputs{1},1)*sc,size(inputs{1},2)*sc,1,size(inputs{1},4)],'like',inputs{1});
            ch = 1;
            for k1 = 1:sc
                for k2 = 1:sc
                    outputs{1}(k1:sc:end,k2:sc:end,:,:) = inputs{1}(:,:,ch,:);
                    ch = ch+1;
                end
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            sc = sqrt(size(inputs{1},3));
            derInputs{1} = inputs{1}*0;
            ch = 1;
            for k1 = 1:sc
                for k2 = 1:sc
                    derInputs{1}(:,:,ch,:) = derOutputs{1}(k1:sc:end,k2:sc:end,:,:);
                    ch = ch+1;
                end
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
        
        function obj = my_ESPC(varargin)
            obj.load(varargin) ;
        end
    end
end
