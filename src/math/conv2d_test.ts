/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array1D, Array2D, Array3D, Array4D} from './ndarray';

// math.conv2d
{
  const tests: MathTests = it => {
    it('input=2x2x1,d2=1,f=1,s=1,p=0', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 1;
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w = Array4D.new([fSize, fSize, inputDepth, outputDepth], [2]);
      const bias = Array1D.new([-1]);

      const result = math.conv2d(x, w, bias, stride, pad);
      const expected = new Float32Array([1, 3, 5, 7]);

      test_util.expectArraysClose(result.getValues(), expected);
      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('input=2x2x1,d2=1,f=1,s=1,p=0,batch=2', math => {
      const inputDepth = 1;
      const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 1;
      const pad = 0;
      const stride = 1;

      const x = Array4D.new(inShape, [1, 2, 3, 4, 5, 6, 7, 8]);
      const w = Array4D.new([fSize, fSize, inputDepth, outputDepth], [2]);
      const bias = Array1D.new([-1]);

      const result = math.conv2d(x, w, bias, stride, pad);
      expect(result.shape).toEqual([2, 2, 2, 1]);
      const expected = new Float32Array([1, 3, 5, 7, 9, 11, 13, 15]);

      test_util.expectArraysClose(result.getValues(), expected);
      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('input=2x2x1,d2=1,f=2,s=1,p=0', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 2;
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w =
          Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
      const bias = Array1D.new([-1]);

      const result = math.conv2d(x, w, bias, stride, pad);
      const expected = new Float32Array([19]);

      test_util.expectArraysClose(result.getValues(), expected);

      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when x is not rank 3', math => {
      const inputDepth = 1;
      const outputDepth = 1;
      const fSize = 2;
      const pad = 0;
      const stride = 1;

      // tslint:disable-next-line:no-any
      const x: any = Array2D.new([2, 2], [1, 2, 3, 4]);
      const w =
          Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
      const bias = Array1D.new([-1]);

      expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when weights is not rank 4', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      // tslint:disable-next-line:no-any
      const w: any = Array3D.new([2, 2, 1], [3, 1, 5, 0]);
      const bias = Array1D.new([-1]);

      expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when biases is not rank 1', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 2;
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w =
          Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
      // tslint:disable-next-line:no-any
      const bias: any = Array2D.new([2, 2], [2, 2, 2, 2]);

      expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when x depth does not match weight depth', math => {
      const inputDepth = 1;
      const wrongInputDepth = 5;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 2;
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w =
          Array4D.randNormal([fSize, fSize, wrongInputDepth, outputDepth]);
      const bias = Array1D.new([-1]);

      expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });
  };

  test_util.describeMathCPU('conv2d', [tests]);
  test_util.describeMathGPU('conv2d', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.conv2d
{
  const tests: MathTests = it => {
    it('conv1d input=2x2x1,d2=1,f=1,s=1,p=same', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 1;
      const pad = 'same';
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w = Array3D.new([fSize, inputDepth, outputDepth], [3]);

      const bias = Array1D.new([0]);

      const result = math.conv1d(x, w, bias, stride, pad);

      const expected = new Float32Array([3, 6, 9, 12]);

      test_util.expectArraysClose(result.getValues(), expected);
      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('conv1d input=1x4x1,d2=1,f=2x1x1,s=1,p=valid', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [1, 4, inputDepth];
      const outputDepth = 1;
      const fSize = 2;
      const pad = 'valid';
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w = Array3D.new([fSize, inputDepth, outputDepth], [2, 1]);

      const bias = Array1D.new([0]);

      const result = math.conv1d(x, w, bias, stride, pad);
      const expected = new Float32Array([4, 7, 10]);

      test_util.expectArraysClose(result.getValues(), expected);
      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('conv1d input=1x4x1,d2=1,f=2x1x1,s=2,p=valid', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [1, 4, inputDepth];
      const outputDepth = 1;
      const fSize = 2;
      const pad = 'valid';
      const stride = 2;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w = Array3D.new([fSize, inputDepth, outputDepth], [2, 1]);

      const bias = Array1D.new([0]);

      const result = math.conv1d(x, w, bias, stride, pad);
      const expected = new Float32Array([4, 10]);

      test_util.expectArraysClose(result.getValues(), expected);
      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when x is not rank 3', math => {
      const inputDepth = 1;
      const outputDepth = 1;
      const fSize = 2;
      const pad = 0;
      const stride = 1;

      // tslint:disable-next-line:no-any
      const x: any = Array2D.new([2, 2], [1, 2, 3, 4]);
      const w =
          Array3D.new([fSize, inputDepth, outputDepth], [3, 1]);
      const bias = Array1D.new([-1]);

      expect(() => math.conv1d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when weights is not rank 3', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      // tslint:disable-next-line:no-any
      const w: any = Array4D.new([2, 2, 1, 1], [3, 1, 5, 0]);
      const bias = Array1D.new([-1]);

      expect(() => math.conv1d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when biases is not rank 1', math => {
      const inputDepth = 1;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 2;
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w =
          Array3D.new([fSize, inputDepth, outputDepth], [3, 1]);
      // tslint:disable-next-line:no-any
      const bias: any = Array2D.new([2, 2], [2, 2, 2, 2]);

      expect(() => math.conv1d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });

    it('throws when x depth does not match weight depth', math => {
      const inputDepth = 1;
      const wrongInputDepth = 5;
      const inputShape: [number, number, number] = [2, 2, inputDepth];
      const outputDepth = 1;
      const fSize = 2;
      const pad = 0;
      const stride = 1;

      const x = Array3D.new(inputShape, [1, 2, 3, 4]);
      const w =
          Array3D.randNormal([fSize, wrongInputDepth, outputDepth]);
      const bias = Array1D.new([-1]);

      expect(() => math.conv1d(x, w, bias, stride, pad)).toThrowError();

      x.dispose();
      w.dispose();
      bias.dispose();
    });
  };

  test_util.describeMathCPU('conv1d', [tests]);
  test_util.describeMathGPU('conv1d', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
