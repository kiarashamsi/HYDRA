"""
Specify the datasets that will be used for training
"""

# Local
DATA_PATH = "Data_Path"
SAVED_MODEL_PATH = "Saved_model_path"



# data pack specification
dataset_pack_dict = {
    "pack_1": [
               ["IOTX"],  # 0
               ["NOIA"],  # 1
               ["QSP"],  # 2
               ["RGT"],  # 3
               ["CMT"],  # 4
               ["Yf-DAI"],  # 5
               ["Mog"],  # 6
               ["FEG"],  # 7
               ["PUSH"],  # 8
               ["HOP"],  # 9
               ["RSR"],  # 10
               ["ORN"],  # 11
               ["SLP"],  # 12
               ["SUPER"],  # 13
               ["ALBT"],  # 14
               ["sILV2"],  # 15
               ["DODO"],  # 16
               ["BOB"],   # 17
               ["GHST"],  # 18
               ["YFII"],  # 19
               ["aDAI"],  # 20
               ["DRGN"],  # 21
               ["CELR"],  # 22
               ["POOH"],  # 23
               ["TVK"],   # 24
               ["PICKLE"],  # 25
               ["LINA"],  # 26
               ["cDAI"],  # 27
               ["ANT"],  # 28
               ["TNT"],  # 29
               ["WOOL"],  # 30
               ["BITCOIN"],  # 31
               ["LQTY"],  # 32
               ["AUDIO"],  # 33
               ["RLB"],  # 34
               ["SWAP"],  # 35
               ["OHM"],  # 36
               ["RARI"],  # 37
               ["REP"],  # 38
               ["LADYS"],  # 39
               ["BTRFLY"],  # 40
               ["bendWETH"],  # 41
               ["TURBO"],  # 42
               ["0x0"],  # 43
               ["PRE"],  # 44
               ["AIOZ"],  # 45
               ["crvUSD"],  # 46
               ["CRU"], # 47
               ["MIM"],  # 48
               ["SPONGE"],  # 49
               ["aUSDC"],  # 50
               ["KP3R"],  # 51
               ["STARL"],  # 52
               ["ShibDoge"],  # 53
               ["LUSD"],  # 54
               ["PSYOP"],  # 55
               ["steCRV"],  # 56
               ["MAHA"],  # 57
               ["POLS"],  # 58
               ["RFD"],  # 59
               ["INU"],  # 60
               ["DPI"],  # 61
               ["ARC"],  # 62
               ["INJ"],  # 63

            #    # test package
            #    ["MIR"],
            #    ["DOGE2.0"],
            #    ["MUTE"],
            #    ["EVERMOON"],
            #    ["DERC"],
            #    ["ADX"],
            #    ["HOICHI"],
            #    ["SDEX"],
            #    ["BAG"],
            #    ["XCN"],
            #    ["ETH2x-FLI"],
            #    ["stkAAVE"],
            #    ["GLM"],
            #    ["QOM"],
            #    ["WOJAK"],
            #    ["DINO"],
            #    ["Metis"],
            #    ["REPv2"],
            #    ["TRAC"],
            #    ["BEPRO"],
               ],  # dataset_package_1

    "pack_2": [["IOTX", "NOIA"]], # dataset_package_2

    "pack_4": [["IOTX", "NOIA", "QSP", "RGT"]], # dataset_package_4

    "pack_8": [["IOTX", "NOIA", "QSP", "RGT", "CMT", "Yf-DAI", "Mog", "FEG"]], # dataset_package_8

    "pack_4_low": [[
        # "bendWETH", "SLP", "steCRV", "crvUSD",
        "BTRFLY", "PUSH", "aDAI", "LUSD"
        ]],

    "pack_4_high": [[
        "FEG",
        "TURBO",
        # "BOB", "POOH",
        "LADYS",
        # "POLS",
        "CMT",
        # "SPONGE"
        ]],

    "pack_16": [["IOTX", "NOIA", "QSP", "RGT", "CMT", "Yf-DAI", "Mog", "FEG",
                "PUSH", "HOP", "RSR", "ORN", "SLP", "SUPER", "ALBT", "sILV2",
                ]], # dataset_package_16

    "pack_32": [["IOTX", "NOIA", "QSP", "RGT", "CMT", "Yf-DAI", "Mog", "FEG",
                "PUSH", "HOP", "RSR", "ORN", "SLP", "SUPER", "ALBT", "sILV2",
                "DODO", "BOB", "GHST", "YFII", "aDAI", "DRGN", "CELR", "POOH",
                "TVK", "PICKLE", "LINA", "cDAI", "ANT", "TNT", "WOOL", "BITCOIN",
                ]], # dataset_package_32

    "pack_64": [["IOTX", "NOIA", "QSP", "RGT", "CMT", "Yf-DAI", "Mog", "FEG",
                "PUSH", "HOP", "RSR", "ORN", "SLP", "SUPER", "ALBT", "sILV2",
                "DODO", "BOB", "GHST", "YFII", "aDAI", "DRGN", "CELR", "POOH",
                "TVK", "PICKLE", "LINA", "cDAI", "ANT", "TNT", "WOOL", "BITCOIN",
                "LQTY", "AUDIO", "RLB", "SWAP", "OHM", "RARI", "REP", "LADYS",
                "BTRFLY", "bendWETH", "TURBO", "0x0", "PRE", "AIOZ", "crvUSD","CRU",
                "MIM", "SPONGE", "aUSDC", "KP3R", "STARL", "ShibDoge", "LUSD", "PSYOP",
                "steCRV", "MAHA", "POLS", "RFD", "INU", "DPI", "ARC", "INJ",
                ]], # dataset_package_64

    "pack_test": [[
        "MIR",
        "DOGE2.0",
        "MUTE",
        "EVERMOON",
        "DERC",
        "ADX",
        "HOICHI",
        "SDEX",
        "BAG",
        "XCN",
        "ETH2x-FLI",
        "stkAAVE",
        "GLM",
        "QOM",
        "WOJAK",
        "DINO",
        "Metis",
        "REPv2",
        "TRAC",
        "BEPRO",
        ]], # dataset_package_test

}
