# coding:  utf-8

'''
領域正規化済み積分 (regular-integration) モジュール
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

EPSILON = 1e-9


def integrateGauss(
        srcMat: np.ndarray,
        foreMask: np.ndarray,
        ky: int,
        kx: int,
        sigmaY: float,
        sigmaX: float,
        regularly: bool=True,
    ):
    '''
    領域正規のガウシアン積分を行う
    '''

    srcMatFloat = srcMat.astype(np.float32)
    foreMaskFloat = foreMask.astype(np.float32)

    summedSrcMat = cv2.GaussianBlur(srcMatFloat, ksize=(kx,ky), sigmaX=sigmaX, sigmaY=sigmaY)
    summedForeMask = cv2.GaussianBlur(foreMaskFloat, ksize=(kx,ky), sigmaX=sigmaX, sigmaY=sigmaY) + EPSILON

    if regularly:
        summedSrcMat = summedSrcMat * foreMaskFloat
        tgtMat = summedSrcMat / summedForeMask
    else:
        tgtMat = summedSrcMat
            
    return tgtMat
    


if __name__ == '__main__':
    '''
    使用例
    '''

    # ------------------------------------------------------------
    # 空間積分対象画像・前景マスクを読み込む
    # ------------------------------------------------------------

    srcMat = cv2.imread(r"sampleData\srcMat.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    foreMask = cv2.imread(r"sampleData\foreMask.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # ------------------------------------------------------------
    # 領域正規の積分を計算
    # ------------------------------------------------------------

    tgtMat = integrateGauss(
        srcMat=srcMat,
        foreMask=foreMask,
        ky=31,
        kx=31,
        sigmaY=15.,
        sigmaX=15.,
        regularly=True,
    )

    # ------------------------------------------------------------
    # 結果表示 (元画像，マスク，領域正規な積分結果，通常の積分結果)
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(8,7))
    ax = plt.subplot(2,2,1)
    im = ax.imshow(srcMat)
    plt.colorbar(im)
    plt.title('src-mat')


    ax = plt.subplot(2,2,2)
    im = ax.imshow(foreMask)
    plt.colorbar(im)
    plt.title('fore-mask')



    ax = plt.subplot(2,2,3)
    im = ax.imshow(tgtMat)
    plt.colorbar(im)
    plt.title('regularly-integrated-mat')


    ax = plt.subplot(2,2,4)
    traditionalIntegrationResult = integrateGauss(
        srcMat=srcMat,
        foreMask=foreMask,
        ky=31,
        kx=31,
        sigmaY=15.,
        sigmaX=15.,
        regularly=False,
    )
    im = ax.imshow(traditionalIntegrationResult)
    plt.colorbar(im)
    plt.title('traditionally-integrated-mat')


    plt.suptitle('The Result of Regular Integration.')
    plt.show()