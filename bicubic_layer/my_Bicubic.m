classdef my_Bicubic < dagnn.Filter
    % Edited by GitChoi
    properties
        size = [0 0 0 0]
        hasBias = false
        opts = {'cuDNN'}
        sc
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            if ~obj.hasBias, params{2} = [] ; end
            outputs{1} = vl_nnconv(...
                inputs{1}, params{1}, params{2}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if ~obj.hasBias, params{2} = [] ; end
            [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
                inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
        end
        
        function kernelSize = getKernelSize(obj)
            kernelSize = obj.size(1:2) ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
            outputSizes{1}(3) = obj.size(4) ;
        end
        
        function params = initParams(obj)
            % loads a bicubic filter (5x5x1xsc^2) for a scaling factor sc.
            aa = load(['bicubic_x',num2str(obj.sc),'_fbic']);
            params{1} = single(aa.fbic);
        end
        
        function set.size(obj, ksize)
            % make sure that ksize has 4 dimensions
            ksize = [ksize(:)' 1 1 1 1] ;
            obj.size = ksize(1:4) ;
        end
        
        function obj = my_Bicubic(varargin)
            obj.load(varargin) ;
            % normalize field by implicitly calling setters defined in
            % dagnn.Filter and here
            obj.size = obj.size ;
            obj.stride = obj.stride ;
            obj.pad = obj.pad ;
        end
    end
end
