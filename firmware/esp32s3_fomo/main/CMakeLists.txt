if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/model/fomo_model_data.cc")
    message(FATAL_ERROR "Thiếu model: chép fomo_model_data.cc, fomo_model_data.h, fomo_settings.h "
                        "(notebook train_fomo_colab.ipynb, mục 8) vào main/model/")
endif()
idf_component_register(SRCS "main.cc" "fomo_decode.cc" "model/fomo_model_data.cc"
                       INCLUDE_DIRS "." "model")
