from segmentation.settings import ColorMap, ClassSpec

if __name__ == "__main__":

    cmap = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                     ClassSpec(label=1, name="Baseline", color=[255, 0, 255]),
                     ClassSpec(label=2, name="BaselineBorder", color=[255, 255, 0])])

    js = cmap.to_json()
    print(js)
