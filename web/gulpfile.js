const gulp = require('gulp');
//const sass=require('gulp-sass');
const browsersync = require('browser-sync').create();
var sass = require('gulp-sass')(require('sass'));

//compile scss into css
function style() {
    return gulp.src('scss/**/*.scss')
        .pipe(sass().on('error', sass.logError)) // Converts Sass to CSS with gulp-sass
        .pipe(gulp.dest('css'))
        .pipe(browsersync.stream());
}

function watch() {
    browsersync.init({
        server: {
            baseDir: './'
        }
    })
    gulp.watch('scss/*.scss', style);
    gulp.watch('scss/partials/*.scss', style);
    gulp.watch('**/*.html').on('change', browsersync.reload);
    gulp.watch('js/**/*.js').on('change', browsersync.reload);
}

exports.style = style;
exports.watch = watch;